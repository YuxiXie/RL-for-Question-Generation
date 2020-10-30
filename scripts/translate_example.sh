# !/bin/bash

set -x

DATAHOME=${HOME}/datasets/process
EXEHOME=${HOME}/src
MODELHOME=${HOME}/models/ensemble

cd ${EXEHOME}

python translate.py \
       -data ${DATAHOME}/process/basic_cased_data_64.pt \
       -model ${MODELHOME}/ensemble.chkpt \
       -output ../predictions/ensemble.txt \
       -gpus 0 \
       -batch_size 32
