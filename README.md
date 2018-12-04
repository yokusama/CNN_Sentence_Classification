# CNN_Sentence_Classification
A Pytorch Implementation of Convolutional Neural Networks for Sentence Classification.
## Introduction

This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch. 


## Requirement
* python 3
* pytorch 0.4
* torchtext 


## Usage
```
python run.py -h 
```
To see the optional arguments

```
optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        CNN rand, static, non-static or multichannel
  --train_data TRAIN_DATA
                        Path of training data
  --dev_data DEV_DATA   Path of dev data
  --test_data TEST_DATA
                        Path of testing data
  --task TASK           SST1 or SST2
  --embedding_name EMBEDDING_NAME
                        Name of pretrained word embedding provided by
                        torchtext
  --embedding_size EMBEDDING_SIZE
                        Pretrained word embedding dimension
  --epoch EPOCH         Maximum training iterations
  --lr LR               Learning rate
  --dropout DROPOUT     Dropout rate
  --batch_size BATCH_SIZE
                        Batch size for one mini-batch
  --seed SEED           Random seed
  --cpu CPU             Use cpu or not
  
```

example:

```
python run.py --task sst1

```
## TO DO

* Finetune the hyper-parameters
* Note : for simplicity, validate and test at the same time by now.
