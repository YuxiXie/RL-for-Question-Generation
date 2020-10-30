# QA-Based Reward for Answerability

Codes for fine-tuning the answerability discriminator based on the SpanBERT-large model [(Joshi et al., 2020)](https://github.com/facebookresearch/SpanBERT). 

## Requirements

#### Environments

`requirements.txt`

#### Data Processing

* We add text _yes no_ at the beginning of each source text to make sure that the answer can be found in the source for each sample.

* We design class `SimpleTokenizer` in `basic_tokenizer.py` for data tokenization

* Datasets splited as train/dev are released in folder [`disriminators/data/answerability`]()

## Training

```bash
bash train.sh
```

## Models

Load fine-tuned answerability discriminator [here]().