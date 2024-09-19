import os.path
import sys
import time
sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    GraceHyperParams
    )
from  easyeditor import (
    CounterFactDataset,
    ZsreDataset,
    CaptionDataset,
    VQADataset,
    WikiRecentDataset,
    KnowEditDataset,
    SanitizationTrainDataset,
    MultiTaskDataset,
    PersonalityDataset,
    SafetyDataset,
    CKnowEditDataset
)
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

from utils import prepare_knowedit_data

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Parameters
    ### Now support ROME only
    parser.add_argument('--editing_method', required=True, type=str)
    ### hparams_dir is the directory of the hyperparameters
    parser.add_argument('--hparams_dir', required=True, type=str)
    ### Default is Sequential Edit
    parser.add_argument('--sequential_edit', default=True, type=bool)

    ## Data
    ### data_type should be the type of the dataset
    parser.add_argument('--data_type', required=True, type=str)
    ### data_dir is the directory of the dataset
    parser.add_argument('--data_dir', required=True, type=str)
    ### exact type of the dataset
    parser.add_argument('--exact_dataset', required=True, type=str)
    ### size of the dataset
    parser.add_argument('--ds_size', default=None, type=int)


    ## Output and logging
    ### metrics_save_dir is the directory to save the metrics
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    ### save_model_dir is the directory to save the edited model
    parser.add_argument('--save_model_dir',default=None, type=str)

    ## Evaluate
    ### To be implemented

    args = parser.parse_args()

    ### Load the hyperparameters
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams

    
    ### Load the dataset
    #### Note: KnowEditDataset contains WikiBio,ZsRE,wiki_counterfact,wiki_recent versions


    dataset_path = os.path.join(args.data_dir, args.exact_dataset)
    print(f'Loading dataset from {dataset_path}')
    if args.data_type == 'zsre':
        dataset = ZsreDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'counterfact':
        dataset = CounterFactDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'caption':
        dataset = CaptionDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'vqa':
        dataset = VQADataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'recent':
        dataset = WikiRecentDataset(dataset_path,size=args.ds_size)
    elif 'knowedit' in args.data_type:
        dataset = KnowEditDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'sanitization':
        dataset = SanitizationTrainDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'multitask':
        dataset = MultiTaskDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'personality':
        dataset = PersonalityDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'safety':
        dataset = SafetyDataset(dataset_path,size=args.ds_size)
    elif args.data_type == 'cknowedit':
        dataset = CKnowEditDataset(dataset_path,size=args.ds_size)
    print(f"Using {args.data_type} dataset")




    ### Prepare the data
    ### if dataset is KnowEditDataset WikiBio,ZsRE,wiki_counterfact,wiki_recent dataset
    if 'knowedit' in args.data_type:
        prompts, subjects, target_new, portability_inputs, locality_inputs = prepare_knowedit_data(dataset)
    else:

        prompts = [dataset_['src'] for dataset_ in dataset]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in dataset]
        target_new = [edit_data_['alt'] for edit_data_ in dataset]
        locality_prompts = [edit_data_['loc'] for edit_data_ in dataset]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in dataset]

        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
        subject = [edit_data_['subject'] for edit_data_ in dataset]

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if args.editing_method == 'IKE':
        train_data_path = '/home/bizon/zns_workspace/24_09_Evaluation/EasyEdit/Method_Test/data/Datasets_for_Factual_Knowledge/data/zsre/zsre_mend_train_10000.json'
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:3')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subjects,
        target_new=target_new,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        sequential_edit=args.sequential_edit,
    )
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)

    if args.save_model_dir is not None:
        save_dir = os.path.join(args.save_model_dir, args.editing_method, time.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        ### note edit args into the save_dir
        json.dump(vars(args), open(os.path.join(save_dir, 'args.json'), 'w'), indent=4)


        
        print(f'Saving model to {save_dir}')
        edited_model.save_pretrained(save_dir, safe_serialization=True)
