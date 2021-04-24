import torch
import pandas as pd
from torch.utils.data import Dataset



class AbstractDataset(Dataset):
    def __init__(self):
        super(AbstractDataset, self).__init__()
        self.args = {"train": None,
                        "target": None,
                        "val" : None}

    def train_preprocess(self, func):
        self.train, args = func(self.train)
        if args: self.args["train"] = args

        assert len(self.train) == len(self.targets)

    def test_preprocess(self, func):
        self.targets, args = func(self.targets)
        if args: self.args["target"] = args

        assert len(self.train) == len(self.targets)

    def __getitem__(self, i):
        #print(self.train[i], self.targets[i])
        return self.train[i], self.targets[i]

    def __len__(self):
        assert len(self.train) == len(self.targets)
        return len(self.train)


class KaggleNLPGettingStarted(AbstractDataset):
    def __init__(self, csv="data/nlp-getting-started/train.csv"):
        super(KaggleNLPGettingStarted, self).__init__()
        self.csv = csv
        self.load()

    def load(self, which_tokenizer="basic_english"):
        df = pd.read_csv(self.csv)
        df.drop(["id", "keyword", "location"], axis=1, inplace=True)
        self.train = df["text"].tolist()

        try:
            self.targets = df["target"].tolist()
        except Exception as e:
            print("Column 'target' not found: ", e)
            self.targets = [0] * len(self.train)

        assert len(self.train) == len(self.targets)

"""
class DailyDialog(AbstractDataset):
    def __init__(self, csv='data/dailydialog/dailydialog.csv'):
        super(DailyDialog, self).__init__()
        self.csv = csv
        self.load()
        self.args = {"train": None,
                        "test": None,
                        "val" : None}

    def load(self):
        df = pd.read_csv(self.csv)
        self.train = df["Text"].tolist()
        self.targets = df["Emotion"].tolist()

        assert len(self.train) == len(self.targets)

"""



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