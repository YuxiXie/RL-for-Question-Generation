#!/bin/bash

set -x

DATAHOME=${HOME}/discriminators/datasets/answerability
EXEHOME=${HOME}/discriminators/src/answerability/code
MODELHOME=${HOME}/discriminators/models/answerability

mkdir -p ${MODELHOME}

export CUDA_VISIBLE_DEVICES=0

python run_mrqa_latest.py \
		--do_train \
		--do_eval \
		--model spanbert-large-cased \
		--train_file ${DATAHOME}/train.txt \
		--dev_file ${DATAHOME}/dev.txt \
		--train_batch_size 32 \
		--eval_batch_size 32 \
		--gradient_accumulation_steps 8 \
		--learning_rate 2e-5 \
		--num_train_epochs 5 \
		--max_seq_length 512 \
		--doc_stride 128 \
		--eval_per_epoch 5 \
		--output_dir ${MODELHOME}

