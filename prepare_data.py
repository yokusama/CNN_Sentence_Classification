# -*- coding: utf-8 -*-
from torchtext import data, datasets
import torchtext
import torch
def prepare_sst1(BATCH_SIZE, DEVICE, VECTORS):
    PAD_TOKEN='<pad>'
    TEXT = data.Field(batch_first=True, lower=True, pad_token=PAD_TOKEN)
    LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

    train_data = data.TabularDataset(path='./data/sst1/sst1_train.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    dev_data = data.TabularDataset(path='./data/sst1/sst1_dev.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    test_data = data.TabularDataset(path='./data/sst1/sst1_test.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    TEXT.build_vocab(train_data, vectors=VECTORS, unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25))
    LABEL.build_vocab(train_data)


    train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE, train=True,
                                     shuffle=True,device=DEVICE)

    dev_iterator = data.Iterator(dev_data, batch_size=len(dev_data), train=False,
                             sort=False, device=DEVICE)

    test_iterator = data.Iterator(test_data, batch_size=len(test_data), train=False,
                              sort=False, device=DEVICE)

    PAD_INDEX = TEXT.vocab.stoi[PAD_TOKEN]
    TEXT.vocab.vectors[PAD_INDEX] = 0.0
    pretrained_embeddings = TEXT.vocab.vectors

    return train_iterator, dev_iterator, test_iterator, pretrained_embeddings, len(TEXT.vocab), len(LABEL.vocab), PAD_INDEX


def prepare_sst2(BATCH_SIZE, DEVICE, VECTORS):
    PAD_TOKEN='<pad>'
    TEXT = data.Field(batch_first=True, lower=True, pad_token=PAD_TOKEN)
    LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

    train_data = data.TabularDataset(path='./data/sst2/sst2_train.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    dev_data = data.TabularDataset(path='./data/sst2/sst2_dev.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    test_data = data.TabularDataset(path='./data/sst2/sst2_test.tsv', format='tsv',
                                fields=[('text', TEXT), ('labels', LABEL)])

    TEXT.build_vocab(train_data, vectors=VECTORS, unk_init= lambda x:torch.nn.init.uniform_(x, a=-0.25, b=0.25))
    LABEL.build_vocab(train_data)


    train_iterator = data.BucketIterator(train_data, batch_size=BATCH_SIZE, train=True,
                                     shuffle=True,device=DEVICE)

    dev_iterator = data.Iterator(dev_data, batch_size=len(dev_data), train=False,
                             sort=False, device=DEVICE)

    test_iterator = data.Iterator(test_data, batch_size=len(test_data), train=False,
                              sort=False, device=DEVICE)

    PAD_INDEX = TEXT.vocab.stoi[PAD_TOKEN]
    TEXT.vocab.vectors[PAD_INDEX] = 0.0
    pretrained_embeddings = TEXT.vocab.vectors

    return train_iterator, dev_iterator, test_iterator, pretrained_embeddings, len(TEXT.vocab), len(LABEL.vocab), PAD_INDEX
