
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import time
import csv

"""
(check) find official dataset and their exact collumn name
"""
class T5Dataset:
    def __init__(self, dataset_name, tokenizer):
        """
        Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        assert dataset_name in ["the-vault-function", "MultiPL-E", "CodeTask-CL"], "Unknown dataset name"
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        if self.dataset_name == "MultiPL-E":
            self.DATASET_REVISION = "3a9e8d226c127392ce81dca29d09f33f0ce3247d"
            self.task_list = ["c_sharp", "cpp", "python", "ruby", "php", "java", "rust", "go", "javascript"]
            self.multiple_subset = {"c_sharp":"cs", "cpp":"cpp", "ruby":"rb", "php":"php", "java":"java", "rust":"rs", "go":"go", "javascript":"js"}
        elif self.dataset_name == "the-vault-function":
            self.task_list = ["c_sharp", "cpp", "python", "ruby", "php", "java", "rust", "go", "javascript", "c"]
            self.text_key = "docstring"
            self.label_key = "code"
            self.train_parquet = {
                "c_sharp":"c_sharp-00000-of-00001.parquet", 
                "cpp":"cpp-00000-of-00001.parquet", 
                "python":"python-00000-of-00002.parquet",
                "ruby":"ruby-00000-of-00001.parquet", 
                "php":"php-00000-of-00001.parquet", 
                "java":"java-00000-of-00002.parquet",
                "rust":"rust-00000-of-00001.parquet", 
                "go":"go-00000-of-00001.parquet",
                "javascript":"javascript-00000-of-00001.parquet",
                "c":"c-00000-of-00001.parquet"
                }
        elif self.dataset_name == "CodeTask-CL":
            self.task_list = ['CodeTrans', 'CodeSearchNet', 'BFP', 'CONCODE']
            self.text_key = {'CONCODE': 'nl',
                            'CodeTrans': 'java',
                            'CodeSearchNet': 'code',
                            'BFP': 'buggy'}
            self.label_key = {'CONCODE': 'code',
                                'CodeTrans': 'cs',
                                'CodeSearchNet': 'docstring',
                                'BFP': 'fixed'}
            self.task_instructions ={ 'CONCODE': 'Generate <language>Java</language> code from the following English description: {}',
                                    'CodeTrans': 'Translate the following <language>Java</language> code <code>{}</code> into <language>C#</language>: ',
                                    'CodeSearchNet': 'The <language>Ruby</language> code <code>{}</code> comment is <comment>',
                                    'BFP': 'Refactor or improve the following <language>Java</language> code <code>{}</code>: '}
            self.dataset_id = {
                'CONCODE': 'AhmedSSoliman/CodeXGLUE-CONCODE',
                'CodeTrans': 'CM/codexglue_codetrans',
                'CodeSearchNet': 'semeru/code-text-ruby',
                'BFP': 'ayeshgk/code_x_glue_cc_code_refinement_annotated'
            }
            
    """
    For code generation tasks: randomly select k examples from the dataset.
    """
    def select_subset_ds(self, ds, k=2000, seed=0):
        np.random.seed(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = np.random.choice(np.arange(ds.shape[0]), num_samples, replace=False)
        return ds.select(idx_total)

    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self,
                            examples,
                            task,
                            max_length=128,
                            max_length_target=128,
                            #batched=False
                            ):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tokenizer = self.tokenizer
        text_key = self.text_key[task] if self.dataset_name == "CodeTask-CL" else self.text_key
        label_key = self.label_key[task] if self.dataset_name == "CodeTask-CL" else self.label_key

        text = examples[text_key].strip()
        if self.dataset_name == "the-vault-function":
            text = f"Write the following function in <language>{task}</language>:" + "\n" + text
        elif self.dataset_name == "CodeTask-CL":
            text = self.task_instructions[task].format(text)

        source = tokenizer(text,
                            padding=False,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                          )

        target_text = examples[label_key].strip()
        target_text += tokenizer.eos_token
        with tokenizer.as_target_tokenizer():
            target = tokenizer(target_text,
                                padding=False,
                                truncation=True,
                                max_length=max_length_target,
                                return_tensors="pt"
                            )
        labels = target["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        dict_final = {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
        return dict_final

    def get_final_ds(self,
                     task,
                     split,
                     batch_size,
                     root_ds_eval=None,
                     k=-1,
                     seed=0,
                     target_len=128,
                     max_length=128,
                     k_test=None):
        """Function that returns final T5 dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            return_test (bool, optional): Whether to create a test split.
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            prefix_list (List[str], optional): List of prompt virtual tokens to pre-pend to the input.
                We do not encode soft prompt as extra virtual tokens in the latest implementation.
                Defaults to [], empty list.

        Returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """
        assert task in self.task_list, f"Unknown task name: {task}"
        if self.dataset_name == "the-vault-function":
            assert split in ["train", "validation", "test"], "split must be train/val/test"
            if split == "train":
                data_files = f"https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/resolve/main/data/train/small/{self.train_parquet[task]}"
            else:
                data_files = f"https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/resolve/main/data/{split}/{task}-00000-of-00001.parquet"
            dataset = load_dataset(
                "parquet",
                data_files=data_files,
                split="train"
            )
        elif self.dataset_name == "MultiPL-E":
          assert split in ["validation", "test"], "split must be val/test"
          assert root_ds_eval is not None, "root_ds_eval must be provided for test split"
          if task == "python":
            dataset = load_dataset(
              "json",
              data_files=f"https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/data/sanitized-mbpp.json",
              split="train"
            )
            dataset = dataset.rename_column("text", "prompt")
          else:
            dataset = load_dataset(
                "nuprl/MultiPL-E", f"{root_ds_eval}-{self.multiple_subset[task]}", revision=self.DATASET_REVISION, split="test"
            )
        elif self.dataset_name == "CodeTask-CL":
            assert split in ["validation", "test", "train"], "split must be val/test/train"
            dataset = load_dataset(
                self.dataset_id[task],
                split=split
            )

        # Selecting k subset of the samples (if requested)
        if k != -1:
            if split == "train":
                dataset = self.select_subset_ds(dataset, k=k)
            else:
                if k_test is not None and k_test != -1: # if k_test is None, use all the val/test data
                    k_sum = k + k_test
                    dataset = self.select_subset_ds(dataset, k=k_sum)
        dataset = dataset.shuffle(seed=seed)

        # Returning the selected data split (train/val/test)
        if split == "train":
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x,
                                                                            task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            ),
                                                                            batched=False,
                                                                            load_from_cache_file=False
                                                                            )
            encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            encoded_dataset = encoded_dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
            return encoded_dataset

        # Creating an extra test set from the selected data split
        else:
            N = len(dataset) 
            dataset_val = dataset.select(np.arange(0, k)) # val_size = k_val
            dataset_test = dataset.select(np.arange(k, N)) # test_size = N - k_val
            if self.dataset_name in ["CodeTask-CL", "the-vault-function"]:
                dataloaders_val_test = []
                for dataset in [dataset_val, dataset_test]:
                    encoded_dataset = dataset.map(lambda x: self.preprocess_function(x,
                                                                                    task,
                                                                                    max_length=max_length,
                                                                                    ),
                                                                                    batched=False,
                                                                                    load_from_cache_file=False
                                                                                    )
                    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                    encoded_dataset = encoded_dataset.remove_columns([col for col in dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']])
                    dataloaders_val_test.append(encoded_dataset)

                return dataloaders_val_test
            elif self.dataset_name == "MultiPL-E":
                dataset_val = dataset_val.remove_columns(["doctests", "original", "prompt_terminology"])
                dataset_test = dataset_test.remove_columns(["doctests", "original", "prompt_terminology"])

            return [dataset_val, dataset_test]
        
def get_task_data_dict(train_ds_name, benchmark, task, tokenizer, seq_len, target_len, split_size_dict):
    ''' return:
    - tasks_data_dict: dict(task_name: {'lora_id': int, 'task_index': int, 'train': Dataloader, 'valid':Dataset, 'test':Dataset})
    '''
    task_data_dict = {}
    source_len_codetask = {'CodeTrans':320, 'CodeSearchNet':256, 'BFP':130, 'CONCODE':320}
    target_len_codetask = {'CodeTrans':256, 'CodeSearchNet':128, 'BFP':120, 'CONCODE':150}
    if benchmark == "CodeTask-CL":
        seq_len = source_len_codetask[task]
        target_len = target_len_codetask[task]

    for split in split_size_dict.keys():
        if split == "train":
            data_params = {
                'task': task,
                'batch_size': split_size_dict['train']['batch_size'],
                'max_length': seq_len,
                'target_len': target_len,
            }

            ds_train = T5Dataset(train_ds_name, tokenizer)

            # Load dataloaders
            dataloader_train = ds_train.get_final_ds(**data_params,
                                                k=split_size_dict['train']['size'],
                                                split='train')
            task_data_dict['train'] = dataloader_train

        if split == "valid" or split == "test":
            ds_val_test = T5Dataset(benchmark, tokenizer)

            if benchmark == 'MultiPL-E':
                data_params['batch_size'] = 1
            else:
                data_params['batch_size'] = split_size_dict[split]['batch_size']
            dataset_val, dataset_test = ds_val_test.get_final_ds(
                **data_params,
                root_ds_eval=None,
                k=split_size_dict['valid']['size'] if split_size_dict.get('valid') else 100,
                split='test',
                k_test=split_size_dict['test']['size'] if split_size_dict.get('test') else -1
            )
            if split == "valid":
                task_data_dict['valid'] = dataset_val
            else:
                task_data_dict['test'] = dataset_test
        print(f"Data for task {task} loaded: [{list(task_data_dict.keys())}], target length: {target_len}")

    return task_data_dict

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    # ds = T5Dataset("MultiPL-E", tokenizer)
    # dataset_val, dataset_test = ds.get_final_ds("c_sharp", "validation", 8, "humaneval", k=16)
    # print(dataset_val)
    # print(dataset_test)

    # ds = T5Dataset("the-vault-function", tokenizer)
    # dataloader = ds.get_final_ds("c_sharp", "train", 8, k=50000)
    # print(len(dataloader))
    # print(next(iter(dataloader)).keys())

    ds = T5Dataset("the-vault-function", tokenizer)
    dataloader_val, dataloader_test = ds.get_final_ds("c_sharp", "validation", 8, k=100, k_test=100)
    print(len(dataloader_val))
    print(next(iter(dataloader_val)).keys())
    print(len(dataloader_test))
    print(next(iter(dataloader_test)).keys())