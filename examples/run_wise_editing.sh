python run_wise_editing.py \
  --editing_method=WISE \
  --hparams_dir=../hparams/WISE/llama-7b \
  --data_dir=../data/wise \
  --ds_size=600 \
  --data_type=hallucination \
  --sequential_edit



#python run_wise_editing.py \
#  --editing_method=WISE \
#  --hparams_dir=../hparams/WISE/llama-7b \
#  --data_dir=../data/wise \
#  --ds_size=3 \
#  --data_type=hallucination
##  --sequential_edit