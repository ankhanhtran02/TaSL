# Task Skill Localization and Consolidation (TaSL)
**Update 2025/07:** We are excited to announce the release of [*Recurrent-KIF*](https://github.com/WoodScene/Recurrent_KIF), an improved version of TaSL, accepted to ACL 2025!

Thank you for your interest in our work! This is the original implementation of our ACL 2024 paper, "[TaSL: Continual Dialog State Tracking via Task Skill Localization and Consolidation](https://www.arxiv.org/abs/2408.09857)", and includes the methods proposed in our latest extended work, "[TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning](https://arxiv.org/abs/2408.05200)."

## Local Setup
```
conda create -n TaSL python=3.8
conda activate TaSL
pip install -r requirements.txt
```

## Step 1. Preliminary Preparation
Replace the corresponding files in the Transformers package with `trainer.py` and `trainer_seq2seq.py`, which have modified the source code to add our importance-aware skill localization method.
Change directory to root:
```
cp -i /workspace/TaSL/trainer_seq2seq.py /venv/TaSL/lib/python3.8/site-packages/transformers/trainer_seq2seq.py
```

Export Wandb API key as environment variable:
```
export WANDB_API_KEY="[YOUR API KEY]"
```

## Step 2. Training (TaSL)
To finetune on CodeTask-CL datasets:
```ruby
./scripts/run_train_TaSL_t5_codetask.sh
```