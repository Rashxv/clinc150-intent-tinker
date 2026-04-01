# Project Report Draft Notes

## Title
Intent Classification for Conversational Queries using CLINC150 and Tinker

## Suggested structure

### 1. Problem statement
Explain intent classification, why CLINC150 is suitable, and why high-cardinality classification matters.

### 2. Dataset preparation
- dataset source and size
- preprocessing
- label encoding
- instruction-response conversion
- whether OOS was included

### 3. Method
- base model used
- LoRA setup
- prompt format
- loss function
- validation-based hyperparameter tuning

### 4. Results
- train/validation losses
- test accuracy and macro-F1
- per-intent table
- confusion analysis

### 5. Discussion
- difficult intent pairs
- ambiguity and label overlap
- limitations
- future work (OOS detection, hierarchical intents, retrieval augmentation)
