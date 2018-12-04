import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, model_name, pretrained_embedding, vocab_size, n_labels, padding_idx,
                 n_filters=100, embedding_size=100,  dropout_p=0.5):
        super(CNN, self).__init__()

        self.model_name = model_name
        #self.Ci = 1
        self.n_channels = 1
        self.n_filters = n_filters
        self.n_labels = n_labels


        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx

        if self.model_name == 'rand' :
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        if self.model_name =='static':
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx).from_pretrained(pretrained_embedding, freeze=True)
        if self.model_name =='non-static':
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx).from_pretrained(pretrained_embedding, freeze=False)
        if self.model_name == 'multichannel':
            self.embedding_static = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx).from_pretrained(pretrained_embedding, freeze=True)
            self.embedding_non = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx).from_pretrained(pretrained_embedding, freeze=False)
            self.n_channels = 2

        self.conv1 = nn.Conv2d(self.n_channels, self.n_filters, kernel_size=(3, embedding_size), padding=(2,0))
        self.conv2 = nn.Conv2d(self.n_channels, self.n_filters, kernel_size=(4, embedding_size), padding=(3,0))
        self.conv3 = nn.Conv2d(self.n_channels, self.n_filters, kernel_size=(5, embedding_size), padding=(4,0))

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(n_filters*3, n_labels)

    def conv_and_maxpool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):

        if self.model_name == 'multichannel':
            y = self.embedding_static(x).unsqueeze(1)
            z = self.embedding_static(x).unsqueeze(1)
            x = torch.cat((y,z), dim=1)
        else:
            x = self.embedding(x).unsqueeze(1) # (N, 1, W, D)
        #batch size, 2, sent len, emb dim
        x1 = self.conv_and_maxpool(x, self.conv1)
        x2 = self.conv_and_maxpool(x, self.conv2)
        x3 = self.conv_and_maxpool(x, self.conv3)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.dropout(x)

        logit = self.fc(x)

        return logit
