import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import out_dim_2d_conv

'''
from transformers import BertModel

class LSTMClassifier(nn.ModuleList):
    def __init__(self, batch_size, hidden_dim, layers, max_words):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm_layers = layers
        self.input_size = max_words

        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.lstm_layers,
                                batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                                out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2, 1)


    def forward(self, x):
        #print(x.shape)
        h = torch.zeros((self.lstm_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.lstm_layers, x.size(0), self.hidden_dim))

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # N, vocab, h_dim
        out = self.embedding(x)
        #print("embed",out.shape)
        #print("prerrn",out.shape, h.shape, c.shape)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        #print("postrrn", out.shape)
        out = torch.relu_(self.fc1(out[:,-1,:]))
        #print("postl1", out.shape)
        out = self.dropout(out)
        #print(out.shape)
        out = torch.sigmoid(self.fc2(out))
        #print("postl2", out.shape)

        return out

class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()

        option_name = "bert-case-uncased"
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.lin = torch.nn.Linear(in_features=73, out_features=1)

    def forward(self, text):
        mask = text.clone()
        mask[text > 0] = 1
        print("Masked")
        outputs = self.encoder(text, attention_mask=mask)
        print("Predicted")
        text = outputs[0]
        text = self.lin(text)
        print("Passed to lin")
        return text
'''

class CNNTextClassifier(nn.Module):
    def __init__(self,  sequence_len,
                        embedding_size,
                        out1_channels,
                        out2_channels=5,
                        kernel_1=(1, 1),
                        kernel_2=(1, 1),
                        paddings=(0, 0),
                        dilations=(1, 1),
                        strides=(1, 1)):
        super(CNNTextClassifier, self).__init__()
        self.seq_len = sequence_len
        self.emb_size = embedding_size

        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=out1_channels,
                                kernel_size=kernel_1,
                                stride=strides,
                                padding=paddings,
                                dilation=dilations)
        H, W = out_dim_2d_conv(sequence_len,
                                embedding_size,
                                kernel_1,
                                paddings,
                                dilations,
                                strides)
        #print(H, W, kernel_2)
        self.conv2 = nn.Conv2d(out1_channels,
                                out2_channels,
                                kernel_size=kernel_2)
        
        H, W = out_dim_2d_conv(H, W,
                                kernel_2,
                                paddings,
                                dilations,
                                strides)
        #print(H, W)
        self.lin = nn.Linear(H*W*out2_channels, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        #print(x.shape)
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        
        x = x.reshape(batch_size, -1)
        #print(x.shape)
        x = self.lin(x)
        #print(x.shape)
        x = self.softmax(x)
        return x.squeeze()

class LSTMTextClassifier(nn.Module):
    def __init__(self, embedding_dim, 
                        hidden_dim, 
                        output_dim,
                        n_layers,
                        bidirectional,
                        dropout):
        super(LSTMTextClassifier, self).__init__()          
        
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2, NUM_CLASSES)
        
        self.act = nn.Sigmoid()
        
    def forward(self, text):        
        packed_output, (hidden, cell) = self.lstm(text)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
        
        return outputs

class MLP(nn.Module):
    def __init__(self, seq_len, embeddings, num_classes):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(seq_len * embeddings, 1000)
        self.lin2 = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        return x