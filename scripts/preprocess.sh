#!/bin/bash

set -x

DATAHOME=${HOME}/datasets
EXEHOME=${HOME}/src

cd ${EXEHOME}

python preprocess.py \
       -copy \
       -train_src ${DATAHOME}/train/train.src.txt -train_tgt ${DATAHOME}/train/train.tgt.txt \
       -valid_src ${DATAHOME}/dev/dev.src.txt -valid_tgt ${DATAHOME}/dev/dev.tgt.txt \
       -save_data ${DATAHOME}/process/basic_cased_data_64.pt \
       -src_seq_length 256 -tgt_seq_length 64 \
       -bert_tokenizer bert-base-cased \
       -share_vocab