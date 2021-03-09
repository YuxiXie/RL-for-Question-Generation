# !/bin/bash

set -x

DATAHOME=${RLQGHOME}/datasets/process
EXEHOME=${RLQGHOME}/src
MODELHOME=${RLQGHOME}/models/ensemble

cd ${EXEHOME}

python translate.py \
       -data ${DATAHOME}/process/basic_cased_data_64.pt \
       -model ${MODELHOME}/ensemble.chkpt \
       -output ../predictions/ensemble.txt \
       -gpus 0 \
       -batch_size 32
