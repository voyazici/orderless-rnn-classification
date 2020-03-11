import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataset import category_dict_sequential, category_dict_sequential_inv
import numpy as np
from collections import OrderedDict
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet = torchvision.models.resnet101(True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, 80)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Attention(nn.Module):
    def __init__(self, attention_dim, hidden_size, encoder_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, encoder_out, decoder_hidden):
        # (N, num_pixels, attention_dim)
        encoder_out = self.dropout(encoder_out)
        att1 = self.encoder_att(encoder_out)
        # (N, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (N, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # (N, num_pixels)
        # alpha = self.softmax(att)
        alpha_softmax = self.softmax(att)
        # (N, encoder_dim)
        attention_weighted_encoding = (
            encoder_out * alpha_softmax.unsqueeze(2)).sum(dim=1)
        # return attention_weighted_encoding, alpha
        return attention_weighted_encoding, att

class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size, attention_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attention_size = attention_size
        ############################################
        self.fc_dim = 2048
        self.encoder_dim = 2048
        ############################################
        self.lstm_cell = nn.LSTMCell(self.embed_size + self.encoder_dim,
                                     self.hidden_size, bias=True)
        self.number_classes = 80
        self.vocab_size =  self.number_classes + 3
        self.fc = nn.Linear(self.hidden_size, self.number_classes + 1)
        self.init_h = nn.Linear(self.fc_dim, self.hidden_size)
        self.init_c = nn.Linear(self.fc_dim, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        self.attention = Attention(self.attention_size, self.hidden_size,
                                   self.encoder_dim)
        self.f_beta = nn.Linear(self.hidden_size, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_out, fc_out, labels, label_lengths):
        label_lengths, sort_ind = label_lengths.sort(
            dim=0, descending=True)
        labels = labels[sort_ind]
        batch_size = fc_out.shape[0]

        encoder_out = encoder_out[sort_ind]

        fc_out = fc_out[sort_ind]
        label_lengths = (label_lengths - 1).tolist()

        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)

        # embeddings
        start_word_idx = torch.LongTensor(
            [category_dict_sequential['<start>']] * batch_size).to('cuda')
        embeddings = self.embedding(start_word_idx)

        predictions = torch.zeros(batch_size, max(label_lengths),
                                  self.number_classes + 1).to('cuda')

        h = self.init_h(fc_out)
        c = self.init_c(fc_out)

        for t in range(max(label_lengths)):
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.lstm_cell(torch.cat([embeddings, attention_weighted_encoding],
                                            dim=1), (h, c))
            preds = self.fc(h)
            predictions[:, t, :] = preds
            next_word_idx = torch.max(preds, 1)[1]
            embeddings = self.embedding(next_word_idx)
        return predictions, labels, label_lengths

class Encoder(nn.Module):
    def __init__(self, encoder_weights=None):
        super(Encoder, self).__init__()
        self.net = Net()
        if encoder_weights:
            print "ENCODER PRETRAINED WEIGHTS"
            self.net.load_state_dict(torch.load(encoder_weights))
        else:
            print "ENCODER IMAGENET WEIGHTS"

    def forward(self, x):
        return_tuple = []
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return_tuple.append(x.permute(0, 2, 3, 1))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return_tuple.append(x)
        return return_tuple
