import torch
import torch.nn as nn
from torchtext import data, datasets
from model import *

def train(model, train_iter, dev_iter=None, test_iter=None, lr=0.001, epoch=100, use_cuda=False):
    if use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(epoch):
        model.train()
        total_loss = 0.0
        accuracy = 0.0
        total_correct = 0.0
        total_num = len(train_iter.dataset)
        steps = 0.0

        for batch in train_iter:

            steps = steps + 1
            train_text, train_labels = batch.text, batch.labels

            optimizer.zero_grad()

            logit = model(train_text)
            loss = criterion(logit, train_labels)

            nn.utils.clip_grad_norm_(model.fc.weight, max_norm=3)

            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            correct = (torch.max(logit, dim=1)[1].view(train_labels.size()) == train_labels).sum()
            total_correct = total_correct + correct.item()
        print("Epoch %d  Training average Loss: %f  Training accuracy: %f" %
                    (e, total_loss/steps, total_correct/total_num))

        if dev_iter is not None:
            eval(model, dev_iter)
        if test_iter is not None:
            eval(model, test_iter, True)

def eval(model, data_iter, test=False):

    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_num = len(data_iter.dataset)
    steps = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in data_iter:
        text, labels = batch.text, batch.labels
        steps = steps + 1
        logit = model(text)
        loss = criterion(logit, labels)

        total_loss = total_loss + loss.item()
        correct = (torch.max(logit, dim=1)[1].view(labels.size()) == labels).sum()
        total_correct = total_correct + correct.item()
        #print(correct)
    if test:
        print("Test average Loss: %f  Test accuracy: %f" %
                    (total_loss/steps, total_correct/total_num))
    else :
        print("Dev average Loss: %f  Dev accuracy: %f" %
                    (total_loss/steps, total_correct/total_num))
