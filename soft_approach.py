#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("data/dailydialog/dailydialog.csv")


# In[3]:


df


# In[4]:


NUM_CLASSES = len(df.Emotion.value_counts())
NUM_CLASSES


# In[5]:


#df = df[(df.Emotion != 0) & (df.Emotion != 2) & (df.Emotion != 3) ]
#df = df.reset_index()


# In[6]:


import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


# In[7]:


df.Text = df.Text.apply(clean_text)
df.Text = df.Text.apply(clean_numbers)


# In[8]:


df


# In[9]:


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have"}


def replace_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

    def replace(match):
        return contraction_dict[match.group(0)]

    def replacer(text):
        return contraction_re.sub(replace, text)

    return replacer


replacer = replace_contractions(contraction_dict)


# In[10]:


df.Text = df.Text.apply(replacer)


# In[11]:


df


# In[12]:


from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')

df.Text = df.Text.apply(tokenizer)


# In[13]:


df


# In[14]:


from nltk.stem import SnowballStemmer

def stemmer(textroll):
    stmmr = SnowballStemmer("english")
    return [stmmr.stem(w) for w in textroll]

df.Text = df.Text.apply(stemmer)


# In[15]:


df


# In[16]:


def purge(textroll):
    return [ w for w in textroll if not w in {"", None, " ", ".", ","} ]

df.Text = df.Text.apply(purge)


# In[17]:


dfs = df[(df.Emotion != 0)]
dfs = dfs.reset_index()


# In[18]:


MAX_LEN = 50
x = df.Text.apply(len)
x_p = dfs.Text.apply(len)


# In[19]:


def padd_cut(max_len, padd_token="[PAD]"):
    def wrapped(text):
        if len(text) >= max_len:
            return text[:max_len]
        text += [padd_token] * (max_len - len(text))
        return text
    return wrapped

df.Text = df.Text.apply(padd_cut(MAX_LEN))
dfs.Text = dfs.Text.apply(padd_cut(MAX_LEN))


# In[20]:


df


# In[21]:


dfs


# In[22]:


import seaborn as sns
sns.displot(x)


# In[23]:


sns.displot(x_p)


# In[24]:


from torchtext.vocab import GloVe
EMBEDDING_LEN = 100
embedding_glove = GloVe(name="6B", dim=EMBEDDING_LEN)


# In[25]:


embedding_glove["[PAD]"]


# In[26]:


import torch
from torch.utils.data import Dataset, DataLoader

class DaisyDialog(Dataset):
    def __init__(self, df, embedding):
        self.df = df
        self.embedding = embedding

    def __getitem__(self, index):

        sentence = self.df.Text[index]
        sentence = [self.embedding[w] for w in sentence]
        sentence = torch.stack(sentence)

        label = self.df.Emotion[index]
        return sentence, torch.tensor(label)

    def __len__(self):
        return len(self.df)


# In[27]:


print(len(df))
print(len(dfs))

dataset = DaisyDialog(df, embedding_glove)
dataset_s = DaisyDialog(dfs, embedding_glove)

print(len(dataset))
print(len(dataset_s))


# In[28]:


def out_dim_2d_conv(H_in,
                    W_in,
                    kernels,
                    paddings=(0, 0),
                    dilations=(1, 1),
                    strides=(1, 1)):
    
    assert len(kernels) == 2
    assert len(paddings) == 2
    assert len(dilations) == 2
    assert len(strides) == 2

    def num(H_in, padding, dilation, kernel):
        return H_in + 2 * padding - dilation * (kernel - 1) - 1

    H_num = num(H_in, paddings[0], dilations[0], kernels[0])
    H_den = strides[0]

    W_num = num(W_in, paddings[1], dilations[1], kernels[1])
    W_den = strides[1]

    return int(H_num/H_den) + 1, int(W_num/W_den) +1


# In[29]:


import torch.nn as nn

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


# In[30]:


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


# In[31]:



class MLP(nn.Module):
    def __init__(self, seq_len, embeddings):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(seq_len * embeddings, 1000)
        self.lin2 = nn.Linear(1000, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        return x


# In[32]:


import torch.optim as O
from tqdm import tqdm

EPOCHS = 10
BATCH_SIZE = 500
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")
#model = CNNTextClassifier(MAX_LEN, EMBEDDING_LEN, 1, kernel_1=(7, 7), kernel_2=(3, 3)).to(DEVICE)
#model = LSTMTextClassifier(EMBEDDING_LEN, 200, 1, 2, True, 0.2).to(DEVICE)
model = MLP(MAX_LEN, EMBEDDING_LEN).to(DEVICE)
print(len(dataset))
print(len(dataset_s))
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataloader = DataLoader(dataset_s, batch_size=BATCH_SIZE)
optimizer = O.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# In[34]:


for e in range(100):
    corrects = 0
    total = 0
    running_loss = 0
    i = 1
    for sentence, label in tqdm(dataloader):
        sentence = sentence.to(DEVICE)
        label = label.to(DEVICE)

        pred = model(sentence)

        # print(sentence.shape, sentence.dtype)
        # print(pred.shape, pred.dtype)
        # print(label.shape, label.dtype)

        pred = pred.squeeze()
        loss = criterion(pred, label)

        y_hat = torch.argmax(pred,dim=1)

        running_loss += loss
        corrects += torch.sum(y_hat == label).item()
        total += len(y_hat)
        i += 1
        #print((running_loss / (BATCH_SIZE * i)).item())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    print()
    print(f"Epoch {e}/{EPOCHS}" )
    #print((running_loss / (BATCH_SIZE * i)).item())
    print(f"Accuracy: {corrects/total}")

