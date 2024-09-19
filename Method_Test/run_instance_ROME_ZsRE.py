import os
import os.path
import sys
sys.path.append('..')

import safetensors
from safetensors.torch import save_model, save_file
import json
import argparse
import random
import torch
from easyeditor import (
    BaseEditor,
    ROMEHyperParams,
    ZsreDataset,
    KnowEditDataset,
    
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default="Rome", type=str) # Rome
    parser.add_argument('--hparams_dir', required=True, type=str) # hparams
    parser.add_argument('--data_dir', required=True, type=str) # data to edit
    parser.add_argument('--ds_size', default=None, type=int) # size of data to edit
    parser.add_argument('--metrics_save_dir', default='./output', type=str) # save metrics
    parser.add_argument('--datatype', default="ZsRE",type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
    parser.add_argument('--model_save_dir', default='./edited_models', type=str)

    args = parser.parse_args()

    ### load the dataset
    # with open(os.path.join(args.data_dir,'benchmark_ZsRE_ZsRE-test-all.json'), 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)

    datas=KnowEditDataset(args.data_dir,size=args.ds_size)



    

    prompts=[data['prompt'] for data in datas]
    subjects=[data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    
    portability_r =[data['portability_r'] for data in datas]
    portability_s =[data['portability_s'] for data in datas]
    portability_l =[data['portability_l'] for data in datas]

    portability_reasoning_prompts=[]
    portability_reasoning_ans=[]
    portability_Logical_Generalization_prompts=[]
    portability_Logical_Generalization_ans=[]
    portability_Subject_Aliasing_prompts=[]
    portability_Subject_Aliasing_ans=[]
    
    portability_data = [portability_r,portability_s,portability_l]
    portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
    portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
    for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
        for item in data:
            if item is None:
                portable_prompts.append(None)
                portable_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                portable_prompts.append(temp_prompts)
                portable_answers.append(temp_answers)
    assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
    
    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts=[]
    locality_Relation_Specificity_ans=[]
    locality_Forgetfulness_prompts=[]        
    locality_Forgetfulness_ans=[]
    
    locality_data = [locality_rs, locality_f]
    locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
    locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
    for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
    locality_inputs = {}
    portability_inputs = {}
    
    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        },
        'Forgetfulness':{
            'prompt':locality_Forgetfulness_prompts,
            'ground_truth':locality_Forgetfulness_ans
        }
    }
    portability_inputs = {
        'Subject_Aliasing':{
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        },
        'reasoning':{
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans           
        },
        'Logical_Generalization':{
            'prompt': portability_Logical_Generalization_prompts,
            'ground_truth': portability_Logical_Generalization_ans           
        }
    }
    hparams=ROMEHyperParams.from_hparams(os.path.join(args.hparams_dir, 'gpt2-xl.yaml'))
    args.pre_file = None    
    print(args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None
    train_ds=None
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

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )



    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    method=args.editing_method
    edit_number=args.ds_size
    datatype=args.datatype


    # 假设 edited_model 是一个 Hugging Face 模型
    save_path = os.path.join(args.model_save_dir, f"test_basic")
    print(f'Saving edited model to {save_path}')

    # 使用 save_pretrained 方法保存模型为 safetensors 格式
    edited_model.save_pretrained(save_path, safe_serialization=True)

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)


from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM



tokenizer.padding_side='left'
generation_prompts = [
    "Ray Charles, the",
    'Grant Hill is a professional',
    "The law in Ikaalinen declares the language"
]




post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    max_length=16,
    max_new_tokens=128
)