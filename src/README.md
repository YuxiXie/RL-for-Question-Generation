# A Package for Neural Question Generation from [WING-NUS](https://wing.comp.nus.edu.sg/)

OpenNQG is a package built on PyTorch from WING-NUS to facilitate Neural Question Generation. It is aimed to provide an integrated toolkit for the QG domain with the following motivation

- **Fair Comparison between Different Methods** - OpenNQG builds an integrated toolkit which contains most mechanisms of current SOTA models of QG tasks

- **Framework of Neural Question Generation Models** - People can develop their own models based on the package

## Mechanisms

Below is an overview of the mechanisms included in OpenNQG

![Mechanism Overview]()

## Installation

1. **Clone from git**

    https://github.com/YuxiXie/OpenNQG.git

2. **Requirements**

    - python3
    - pytorch-pretrained-bert
    - pytorch

## Simple Example

To run a GRU(Encoder)-GRU(Deocder) based model, 

1. **data preprocessing**

    Prepare tokenized input files: `src.txt` , `tgt.txt` ( and `ans.txt` ) with one line for one sample , and put them at `DATA_HOME`

    Then
    ```bash
    cd ${OpenNQG_HOME}/code
    
    python3 preprocess.py \
            -train_src ${DATA_HOME}/data/train/train.src.txt -train_tgt ${DATA_HOME}/data/train.tgt.txt \
            -valid_src ${DATA_HOME}/data/valid/valid.src.txt -valid_tgt ${DATA_HOME}/data/valid.tgt.txt \
            -answer enc -train_ans ${DATA_HOME}/data/train/train.ans.txt -valid_ans ${DATA_HOME}/data/valid/valid.tgt.txt \
            -save_data ${DATA_HOME}/data/preprocessed_data.pt
    ```

    See `pargs.py` to learn more options of OpenNQG data preprocessing

    `pre.run.sh` is an example script with more details

2. **model training**

    ```bash
    cd ${OpenNQG_HOME}/code

    python3 train.py \
            -data ${DATA_HOME}/data/preprocessed_data.pt \
            -answer enc \
            -d_word_vec 256 \
            -d_enc_model 512 -n_enc_layer 1 -brnn -enc_rnn gru \
            -d_dec_model 512 -n_dec_layer 1 -dec_rnn gru \
            -save_model ${MODEL_HOME}/prefix_of_model_name \
            -gpus 1
    ```

    See `xargs.py` to learn more options of OpenNQG model training

     `run.sh` is an example script with more details

3. **result translating**

    ```bash
    cd ${OpenNQG_HOME}/code

    python3 translate.py \
            -model ${MODEL_HOME}/prefix_of_model_name.chkpt \
            -data ${DATA_HOME}/data/preprocessed_data.pt \
            -output ${DATA_HOME}/predict/predict.txt \
            -gpus 1
    ```