
import re
import torch
import pandas as pd
import torch.optim as O

from torch import nn
from tqdm import tqdm
from models import MLP
from torchtext.vocab import GloVe
from mydatasets import DaisyDialog
from torch.utils.data import DataLoader

from utils import padd_cut # move to commons
from torchtext.data.utils import get_tokenizer

from commons import clean_text, clean_numbers, replace_contractions, stemmer, purge



df = pd.read_csv("data/dailydialog/dailydialog.csv")


MAX_LEN = 50
EMBEDDING_LEN = 100
NUM_CLASSES = len(df.Emotion.value_counts())
EPOCHS = 10
BATCH_SIZE = 500
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
SAVE_PATH = "save/model.pt"

df.Text = df.Text.apply(clean_text)
df.Text = df.Text.apply(clean_numbers)

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have"}
replacer = replace_contractions(contraction_dict)
df.Text = df.Text.apply(replacer)

tokenizer = get_tokenizer('basic_english')
df.Text = df.Text.apply(tokenizer)

df.Text = df.Text.apply(stemmer)

df.Text = df.Text.apply(purge)

dfs = df[(df.Emotion != 0)]
dfs = dfs.reset_index()

x = df.Text.apply(len)
x_p = dfs.Text.apply(len)

df.Text = df.Text.apply(padd_cut(MAX_LEN))
dfs.Text = dfs.Text.apply(padd_cut(MAX_LEN))

embedding_glove = GloVe(name="6B", dim=EMBEDDING_LEN)


embedding_glove["[PAD]"]

print("legnth dataset: ", len(df))
print("length purged dataset: ", len(dfs))

dataset = DaisyDialog(df, embedding_glove)
dataset_s = DaisyDialog(dfs, embedding_glove)

print("length loaded dataset: ", len(dataset))
print("leght loaded purged dataset: " , len(dataset_s))

print(f"Running on: {DEVICE}")

#model = CNNTextClassifier(MAX_LEN, EMBEDDING_LEN, 1, kernel_1=(7, 7), kernel_2=(3, 3)).to(DEVICE)
#model = LSTMTextClassifier(EMBEDDING_LEN, 200, 1, 2, True, 0.2).to(DEVICE)
model = MLP(MAX_LEN, EMBEDDING_LEN, NUM_CLASSES).to(DEVICE)

print(len(dataset))
print(len(dataset_s))

#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataloader = DataLoader(dataset_s, batch_size=BATCH_SIZE)
optimizer = O.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# In[34]:


for e in range(1, EPOCHS):
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

torch.save(model.state_dict(), SAVE_PATH)
