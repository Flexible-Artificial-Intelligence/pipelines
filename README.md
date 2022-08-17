![Pipelines logo](assets/pipelines_logo.png)


## TO-DO

### Natural Language Processing

- Data
    - [ ] Dynamic Padding
    - [ ] Uniform Dynamic Padding

#### Masked Language Modeling
#### Whole Word Masking
#### Replaced Token Detection

#### Named Entity Recognition / Token Classification
- Pre-Processing
    - Tagging schemes
        - [ ] IO
        - [ ] BIO
        - [ ] BIEO
    - Augmentations
        - [ ] CutMix
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
 
- Ensembling
    - [ ] Token-wise blending
    - [ ] Word-wise blending
    - [ ] Char-wise blending
    - [ ] Weighted Box Fusion
    - [ ] Stacking

- [ ] Trainer
