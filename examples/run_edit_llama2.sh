#!/bin/bash

ROME 
python run_edit_llama2.py \
    --editing_method ROME \
    --hparams_dir ../hparams/ROME/llama-7b.yaml \
    --sequential_edit True \
    --data_type knowedit_r \
    --data_dir ../KnowEdit/benchmark/ZsRE/ \
    --exact_dataset ZsRE-test-all.json \
    --ds_size 2 \
    --save_model_dir ../edited_models/ \

# MEMIT
# python run_edit_llama2.py \
#     --editing_method MEMIT \
#     --hparams_dir ../hparams/MEMIT/llama-7b.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \



# MEMIT gpt2

# python run_edit_llama2.py \
#     --editing_method MEMIT \
#     --hparams_dir ../hparams/MEMIT/gpt2-xl.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \

# GRACE gpt2

# python run_edit_llama2.py \
#     --editing_method GRACE \
#     --hparams_dir ../hparams/GRACE/gpt2-xl.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \

#FT gpt2
# python run_edit_llama2.py \
#     --editing_method FT \
#     --hparams_dir ../hparams/FT/gpt2-xl.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \


# FT llama2
python run_edit_llama2.py \
    --editing_method FT \
    --hparams_dir ../hparams/FT/llama-7b.yaml \
    --sequential_edit True \
    --data_type knowedit_r \
    --data_dir ../KnowEdit/benchmark/ZsRE/ \
    --exact_dataset ZsRE-test-all.json \
    --ds_size 2 \
    --save_model_dir ../edited_models/ \

# # KN gpt2-xl
# python run_edit_llama2.py \
#     --editing_method KN \
#     --hparams_dir ../hparams/KN/gpt2-xl.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \


# # KN llama2
# python run_edit_llama2.py \
#     --editing_method KN \
#     --hparams_dir ../hparams/KN/llama-7b.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \


# IKE gp2-xl
# python run_edit_llama2.py \
#     --editing_method IKE \
#     --hparams_dir ../hparams/IKE/gpt2-xl.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \

# IKE llama2-7b
# python run_edit_llama2.py \
#     --editing_method IKE \
#     --hparams_dir ../hparams/IKE/llama-7b.yaml \
#     --sequential_edit True \
#     --data_type knowedit_r \
#     --data_dir ../KnowEdit/benchmark/ZsRE/ \
#     --exact_dataset ZsRE-test-all.json \
#     --ds_size 2 \
#     --save_model_dir ../edited_models/ \
