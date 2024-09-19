import os
import logging
import sys
sys.path.append('..')
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
USE_DEVICE = f"cuda:1"
logging.info(f"Use device: {USE_DEVICE}")

prompts = ['Ray Charles, the',
            'Grant Hill is a professional',
            'The law in Ikaalinen declares the language'
            ]
ground_truth = ['piano',
                'basketball',
                'Finnish'
                ]
target_new = ['violin',
              'soccer',
              'Swedish'
              ]
subject = ['Ray Charles',
            'Grant Hill',
            'Ikaalinen'
            ]



hparams = ROMEHyperParams.from_hparams(os.path.join(PROJECT_PATH, '../hparams/ROME/gpt2-xl.yaml'))
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
)

print(metrics)


print('*'*20)


model_path = "../hugging_cache/gpt2-xl"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, eos_token='<|endoftext|>', pad_token='<|endoftext|>', unk_token='<|endoftext|>')

tokenizer.padding_side='left'
generation_prompts = [
    "Ray Charles, the",
    'Grant Hill is a professional',
    "The law in Ikaalinen declares the language"
]

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(USE_DEVICE)
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to(USE_DEVICE),
    attention_mask=batch['attention_mask'].to(USE_DEVICE),
    max_length=16,
    max_new_tokens=128
)

post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to(USE_DEVICE),
    attention_mask=batch['attention_mask'].to(USE_DEVICE),
    max_length=16,
    max_new_tokens=128
)

pre_edit_outpts = [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
post_edit_outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]

for pre_edit_outpt, post_edit_output in zip(pre_edit_outpts, post_edit_outputs):
    print('Pre-Edit Output: ', "".join(pre_edit_outpt).replace('<|endoftext|>', "").replace('<|im_start|>', "").replace('<|im_end|>', "").replace('\n', ""))
    print('Post-Edit Output: ', "".join(post_edit_output).replace('<|endoftext|>', "").replace('<|im_start|>', "").replace('<|im_end|>', "").replace('\n', ""))