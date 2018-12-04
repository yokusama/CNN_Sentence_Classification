from model import *
from prepare_data import *
from trainer import *

import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Train A CNN ")

    parser.add_argument("--model_name", type=str, default="rand",
                        help="CNN rand, static, non-static or  multichannel")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path of training data")
    parser.add_argument("--dev_data", type=str, default=None,
                        help="Path of dev data")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path of testing data")
    parser.add_argument("--task", type=str, default="sst1",
                        help="SST1 or SST2")
    parser.add_argument("--embedding_name", type=str,default="glove.6B.100d",
                        help="Name of pretrained word embedding provided by torchtext")
    parser.add_argument("--embedding_size", type=int, default=100,
                        help="Pretrained word embedding dimension")
    parser.add_argument('--epoch', type=int, default=100,
                        help="Maximum training iterations")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for one mini-batch")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--cpu", type=bool,default= False,
                        help="Use cpu or not ")

    return parser.parse_args()


params = parse_args()

if params.cpu :
    USE_CUDA = False
    DEVICE = torch.device('cpu')
else:
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if params.task == 'sst1':
    train_iterator, dev_iterator, test_iterator, pretrained_embeddings,len_vocab, n_labels, pad_index = prepare_sst1(params.batch_size, DEVICE, params.embedding_name)

if params.task == 'sst2':
    train_iterator, dev_iterator, test_iterator, pretrained_embeddings,len_vocab, n_labels, pad_index = prepare_sst2(params.batch_size, DEVICE, params.embedding_name)


model = CNN(params.model_name, pretrained_embeddings, len_vocab, n_labels, pad_index, embedding_size=params.embedding_size, dropout_p=params.dropout)

train(model, train_iterator, dev_iterator, test_iterator, params.lr, params.epoch, USE_CUDA)
