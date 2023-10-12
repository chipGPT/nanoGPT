
# tiny stories
Users who prefer to upload a dataset programmatically can use the huggingface_hub library. This library allows users to interact with the Hub from Python.

Plz do follwing before run prepare.py or prepare_tiktoken.py
Begin by installing the library:
```
pip install huggingface_hub
```
To upload a dataset on the Hub in Python, you need to log in to your Hugging Face account:
```
huggingface-cli login
```
you will be asked for choosing model, 0 for little Shakespeare, 1 for tinystories

clear generated files when rerun prepare.py or prepare_tiktoken.py

when run config/train_shakespeare_char.py remember to change the following params
```
wandb_project = 'experiments'
wandb_run_name = 'mini-gpt'
dataset = 'experiments'
```

