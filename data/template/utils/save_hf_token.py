from getpass import getpass
from huggingface_hub import HfFolder

HF_TOKEN = getpass("Enter your Hugging Face token: ")

HfFolder.save_token(HF_TOKEN) 
print("Token saved successfully!")
