import transformers
import os

### make dir
os.makedirs('hugging_cache', exist_ok=True)

### download model
models=["meta-llama/Llama-2-7b-hf","mistralai/Mistral-7B-v0.1"]
TOKEN='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'

for model in models:
    print(f"Downloading {model}")
    model_name=model.split("/")[-1]
    transformers.AutoModel.from_pretrained(model,force_download=True,use_auth_token=TOKEN,cache_dir=f"hugging_cache/{model_name}")
    transformers.AutoTokenizer.from_pretrained(model,force_download=True,use_auth_token=TOKEN,cache_dir=f"hugging_cache/{model_name}")