from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np
import os
import re

#os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import sys
from peft import PeftModel
import wandb
import fire
from utils.prompter import Prompter
from utils.dataset_order import get_dataset_order
import argparse
from transformers import set_seed
from utils.lora_importance_T5 import RankAllocator
from t5_dataset import get_task_data_dict
set_seed(42)
import smooth_bleu
from bleu import _bleu

wandb.login(key=os.environ['WANDB_API_KEY'])


def get_last_checkpoint_path(output_dir: str):
    """
    Finds the path to the last saved checkpoint directory within a given output directory.
    
    This function scans the directory for subdirectories named in the Hugging Face 
    Trainer format, typically 'checkpoint-N' where N is a step number, and returns 
    the path to the one with the highest step number.

    Args:
        output_dir: The main output directory where training checkpoints are saved.

    Returns:
        The full path to the last checkpoint directory, or None if no checkpoints are found.
    """
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found at {output_dir}")
        return None

    # Regex to match checkpoint directories, e.g., 'checkpoint-1000'
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    
    checkpoints = []
    max_step = -1

    try:
        # Iterate over all items in the output directory
        for item in os.listdir(output_dir):
            match = checkpoint_pattern.match(item)
            
            if match and os.path.isdir(os.path.join(output_dir, item)):
                # Extract the step number (the part matched by (\d+))
                step = int(match.group(1))
                
                if step > max_step:
                    max_step = step
                    # Store the full directory name, not just the step number
                    last_checkpoint_dir = item
                    
        if max_step >= 0:
            # Construct and return the full path to the last checkpoint
            full_path = os.path.join(output_dir, last_checkpoint_dir)
            print(f"Found last checkpoint: {full_path}")
            return full_path
        else:
            print(f"No checkpoint directories found in {output_dir}.")
            return None

    except Exception as e:
        print(f"An error occurred while scanning the directory: {e}")
        return None

def main(args):
    print("laile")
    dataset_order = args.task_list
    service_id = args.service_begin_id
    task = dataset_order[service_id]
    model_name = args.model_path.split("/")[-1]

    wandb.init(
        project="CL4Code", # Set your project name here
        group="TaSL",       # Set your group name here
        name=model_name+"-"+"CodeTask-CL"+"-"+str(service_id)+"-"+task,
        reinit=True # Allows running the function multiple times in the same session
    )

    checkpoint_dir = os.path.join("./checkpoint_files", model_name+"-"+"CodeTask-CL", str(service_id) + "-" + task)
    print(f"output_dir: {checkpoint_dir}")

    if service_id == 0:
        resume_from_checkpoint = None
    else:
        last_service_name = dataset_order[service_id - 1]
        last_checkpoint_dir = os.path.join("./checkpoint_files", model_name + "-" + "CodeTask-CL",  str(service_id - 1) + "-" + last_service_name)
        last_checkpoint_dir = get_last_checkpoint_path(last_checkpoint_dir)
        resume_from_checkpoint = last_checkpoint_dir

        if os.path.exists(resume_from_checkpoint):
            print(f"Restarting from {resume_from_checkpoint}")
        else:
            print(f"resume_from_checkpoint dir {resume_from_checkpoint} not find!")
            sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
    

    task_data_dict = get_task_data_dict(
        train_ds_name="CodeTask-CL",
        benchmark="CodeTask-CL",
        task=task,
        tokenizer=tokenizer,
        seq_len=max_input_length,
        target_len=max_target_length,
        split_size_dict={
            "train": {"size": args.train_size, "batch_size": args.train_batch_size},
            "valid": {"size": args.val_size, "batch_size": args.eval_batch_size},
        }
    )
    train_data = task_data_dict['train']
    val_data = task_data_dict['valid']
    print(f"train_data: {train_data}")
    print(f"val_data: {val_data}")    

    output_dir = os.path.join("./outputs", task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"create {output_dir} folder")
    output_fn = os.path.join(output_dir, "val.predictions")
    gold_fn = os.path.join(output_dir, "val.gold")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions = []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for id, (pred_nl, gold) in enumerate(zip(decoded_preds, decoded_labels)):
                if task in ['CodeSearchNet']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(id) + '\t' + pred_nl)
                    f.write(str(id) + '\t' + repr(pred_nl.strip()) + '\n')
                    f1.write(str(id) + '\t' + repr(gold.strip()) + '\n')
                else:
                    f.write(repr(pred_nl.strip()) + '\n')
                    f1.write(repr(gold.strip()) + '\n')
        if task == 'CodeSearchNet':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
        result = {"bleu": bleu,}
        return result

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    # print(model)
    # sys.exit(1)
    
    model_name = args.model_path.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy = "steps",
        save_strategy="steps",
        learning_rate = 3e-4,
        warmup_steps=50,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        weight_decay = 0.01,
        save_total_limit =2,
        load_best_model_at_end=True,
        eval_steps=5,  # 500
        save_steps=5,  # 500
        output_dir=checkpoint_dir,
        num_train_epochs = args.num_epochs,
        predict_with_generate = True,
        # fp16 = True,
        push_to_hub = False,
        # logging_dir=log_dir,
        logging_steps=5, # 10
        report_to=["wandb"],
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )
    
    # new
    rankallocator = RankAllocator(
        model,
        init_warmup=50,
        beta1=args.beta1, 
        beta2=args.beta2, 
    )
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        ipt_score = rankallocator,
        train_dataset = train_data,
        eval_dataset = val_data,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    #resume_from_checkpoint = None
    train_result = trainer.train()

    ipt_name_list, ipt_score_list = rankallocator.calculate_score(metric="ipt")    
    print(ipt_name_list)
    print(ipt_score_list)
    
    if np.isnan(ipt_score_list).any():
        raise ValueError("important score NaN ")
    
    data = {'Module_Name': ipt_name_list, 'Importance_Score': ipt_score_list}
    df = pd.DataFrame(data)


    if service_id == 0:
        csv_file_path = "./ipt_file/"+ model_name+ "_Importance_Score_averaging_dataset_CodeTask-CL_" + str(service_id) + "-" + task + ".csv"
    else:
        csv_file_path = "./ipt_file/"+ model_name +"_Importance_Score_dataset_CodeTask-CL_"+ str(service_id) + "-" + task + ".csv"
    if not os.path.exists(csv_file_path):
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    df.to_csv(csv_file_path, index=False)

    model.save_pretrained(checkpoint_dir)
    
    # df_log = pd.DataFrame(trainer.state.log_history)
    #df_log.to_csv(os.path.join(log_dir,"train_log.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool)

    parser.add_argument("--model_path", type=str, default="Salesforce/codet5-small", help="Path to the base model")
    parser.add_argument("--service_begin_id", type=int, default=0, help="Starting service ID for continual learning")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    
    parser.add_argument("--beta1", type=float, default=0.85, help="Beta1 parameter for importance allocation")
    parser.add_argument("--beta2", type=float, default=0.85, help="Beta2 parameter for importance allocation")

    parser.add_argument("--max_input_length", type=int, default=320, help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=256, help="Maximum target sequence length")
    
    parser.add_argument("--task_list", nargs="+", type=str, default=[], help="List of tasks for continual learning (provide one or more tasks)")
    parser.add_argument("--train_size", type=int, default=-1, help="Training dataset size")
    parser.add_argument("--val_size", type=int, default=100, help="Validation dataset size")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size per device")

    args = parser.parse_args()
    print(
        f"Training T5 model with params:\n"
        f"service_begin_id: {args.service_begin_id}\n"
        f"base_model: {args.model_path}\n"
        f"beta1: {args.beta1}\n"
        f"beta2: {args.beta2}\n"
        f"train_batch_size: {args.train_batch_size}\n"
        f"eval_batch_size: {args.eval_batch_size}\n"
        f"num_epochs: {args.num_epochs}\n"
        f"max_input_length: {args.max_input_length}\n"
        f"max_target_length: {args.max_target_length}\n"
        f"task_list: {args.task_list}\n"
        f"train_size: {args.train_size}\n"
        f"val_size: {args.val_size}\n"
    )
    main(args)
