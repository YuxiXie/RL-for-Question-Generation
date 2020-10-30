# LM-based Reward for Fluency

Codes for fine-tuning the fluency discriminator based on the BERT-base model [(huggingface/transformers)](https://github.com/huggingface/transformers). 

## Requirements

#### Environments

```
transformers 2.7.0
pytorch 1.4.0
nltk 3.4.4
numpy 1.18.1
tqdm 4.32.2
```

#### Data Processing

* Datasets splited as train/dev are released in folder [`disriminators/data/fluency`]()

## Training

```bash
bash train.sh
```

## Models

Load fine-tuned fluency discriminator [here]().