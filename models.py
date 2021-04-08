import torch
import torch.nn as nn
import torch.nn.functional as F

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


