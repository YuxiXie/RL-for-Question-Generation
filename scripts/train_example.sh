# !/bin/bash

set -x

DATAHOME=${HOME}/datasets/process
EXEHOME=${HOME}/src
MODELHOME=${HOME}/models/ensemble
LOGHOME=${HOME}/models/ensemble/logs
RLMODELHOME=${HOME}/discriminators/models

mkdir -p ${MODELHOME}
mkdir -p ${LOGHOME}

cd ${EXEHOME}

python train.py \
       -gpus 0 -rl_gpu 1 1 1 \
       -data ${DATAHOME}/basic_cased_data_64.pt \
       -checkpoint ${HOME}/models/baseline/baseline_cased.chkpt \
       -rl fluency relevance answerability \
       -rl_model_dir ${RLMODELHOME}/fluency ${RLMODELHOME}/relevance ${RLMODELHOME}/answerability \
       -epoch 100 -batch_size 64 -eval_batch_size 32 \
       -max_token_src_len 256 -max_token_tgt_len 64 \
       -copy -coverage -coverage_weight 0.4 \
       -d_word_vec 300 \
       -d_enc_model 512 -n_enc_layer 1 -brnn -enc_rnn gru \
       -d_dec_model 512 -n_dec_layer 1 -dec_rnn gru -d_k 64 \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.3 -attn_dropout 0.1 \
       -save_mode best -save_model ${MODELHOME}/ensemble \
       -logfile_train ${LOGHOME}/ensemble.train \
       -logfile_dev ${LOGHOME}/ensemble.dev \
       -log_home ${LOGHOME} \
       -translate_ppl 20 \
       -curriculum 0  -extra_shuffle -optim adam -learning_rate 0.00001 -learning_rate_decay 0.75 \
       -valid_steps 250 -decay_steps 250 -start_decay_steps 5000 -decay_bad_cnt 5 -max_grad_norm 5 -max_weight_value 32 
