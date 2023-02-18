# PyTorch Lightning Pipelines


## Skeleton

```py
from pytorch_lightning_pipelines.models import ClassificationModel
from pytorch_lightning_pipelines.datasets import ClassificationDataset
from pytorch_lightning import Trainer
from torch.data.utils import DataLoader


# config
pretrained_model_name_or_path = "bert-base-uncased"

# data
texts = ["I'm happy", "I'm angry"]
labels = [1, 0]

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

# dataset
dataset = ClassificationDataset(
    texts=texts, 
    labels=labels, 
    tokenizer=tokenizer, 
    max_length="tokenizer",
)

# dataloader
dataloader = DataLoader(
    dataset=dataset, 
    batch_size=16, 
    drop_last=True, 
    shuffle=True, 
)

# model
model = ClassificationModel(
    pretrained_model_name_or_path=pretrained_model_name_or_path, 
    losses=["cross_entropy"],
    metrics=["accuracy", "f1"],
)


# training
trainer = Trainer(...)
trainer.fit(model, train_dataloaders=[dataloader], ckpt_path=None)
```

## Tasks

#### Binary classification

#### Multi-class classification

#### Multi-label classification

#### Regression

#### Ordinal Regression

#### Named Entity Recognition

#### Question & Answering 

#### Retrieval

#### Language Modeling

#### Causal Language Modeling

https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

#### Masked Language Modeling

https://arxiv.org/abs/1810.04805

#### Sentence Order Prediction

https://arxiv.org/abs/1810.04805

#### Whole Word Masking

https://arxiv.org/abs/1906.08101

#### Permutation Language Modeling

 https://arxiv.org/abs/2203.06906

#### Replaced Token Detection 

 https://arxiv.org/abs/2003.10555

#### Predicting Spans

https://arxiv.org/abs/1907.10529

#### Language Modeling alternatives

https://arxiv.org/abs/2109.01819


### Additional literature

Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks - https://arxiv.org/abs/2004.10964

Masked and Permuted Pre-training for Language Understanding - https://arxiv.org/abs/2004.09297

Self-training Improves Pre-training for Natural Language Understanding  - https://arxiv.org/abs/2010.02194

Should You Mask 15% in Masked Language Modeling?  - https://arxiv.org/abs/2202.08005

## Features

#### Optimization approaches

#### Sliding window

#### Curriculum Learning

#### HuggingFace integration

#### Configuration support

#### Large-scale models support

#### Wide range of losses and metrics

#### Multi-task Learning support / Auxiliary Target Learning

<!-- - Pre-Processing
    - Tagging schemes
        - [ ] IO
        - [ ] BIO
        - [ ] BIEO
    - Augmentations
        - [ ] CutMix
        - [ ] MixUp
        - [ ] Masking
        - [ ] Removing context
- Losses
    - [ ] Cross-Entropy
    - [ ] Weighted Cross-Entropy
    - [ ] Focal Loss
    - [ ] Jaccard Index
    - [ ] Lovasz Loss
    - [ ] Dice Loss
    - [ ] Combinations of losses
    - [ ] + Label Smoothing

- Post-Procsssing
    - [ ] Correction over first and zero tokens
    - [ ] Converting "middle" entities to the neighbors
    - [ ] Filtering entities with lengths and confidence scores
    - [ ] Adding some tokens to increase span's length
    - [ ] Beam Search
    - [ ] Conditional Random Fields
 
- Ensembling
    - [ ] Token-wise blending
    - [ ] Word-wise blending
    - [ ] Char-wise blending
    - [ ] Weighted Box Fusion
    - [ ] Stacking -->
