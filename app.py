import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from utils import token_idx, padder
from models import LSTMClassifier
from datasets import KaggleNLPGettingStarted



BATCH_SIZE = 100
EPOCHS = 5

train_set = KaggleNLPGettingStarted()
train_set.train_preprocess(token_idx)
train_set.train_preprocess(padder(10, 0))


test_set = KaggleNLPGettingStarted(csv="data/nlp-getting-started/test.csv")
test_set.train_preprocess(token_idx)
test_set.train_preprocess(padder(10, 0))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)



model = LSTMClassifier(batch_size=BATCH_SIZE,
                        hidden_dim=100,
                        layers=2,
                        max_words= train_set.args["train"]["len"])

optimizer = optim.RMSprop(model.parameters())

for epoch in range(EPOCHS):
    predictions = []

    model.train()

    for x_batch, y_batch in train_loader:
        x = torch.stack(x_batch).permute(1, 0)
        y = y_batch.type(torch.float)

        y_hat = model(x).squeeze()

        loss = F.binary_cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
predictions = []

model.eval()

with torch.no_grad():
    for x_batch, _ in test_loader:
        x = torch.stack(x_batch).permute(1, 0)

        y_pred = model(x)

        predictions += list(y_pred.detach().numpy())

