# Discriminator-Based Reward for Relevance

Codes for fine-tuning the relevance discriminator based on the BERT-base model [(huggingface/transformers)](https://github.com/huggingface/transformers). 

#### Environments

```
transformers 2.7.0
pytorch 1.4.0
nltk 3.4.4
numpy 1.18.1
tqdm 4.32.2
```

#### Data Processing

* Datasets splited as train/dev are released in folder [`disriminators/data/relevance`]()

## Training

```bash
bash train.sh
```

## Models

Load fine-tuned relevance discriminator [here]().