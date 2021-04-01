from torch.utils.data import Dataset
import pandas as pd

class KaggleNLPGettingStarted(Dataset):
    def __init__(self, csv="data/nlp-getting-started/train.csv"):
        super(KaggleNLPGettingStarted, self).__init__()
        self.csv = csv
        self.load()
        self.args = {"train": None,
                        "test": None,
                        "val" : None}


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

    def train_preprocess(self, func):
        self.train, args = func(self.train)
        if args: self.args["train"] = args

        assert len(self.train) == len(self.targets)

    def test_preprocess(self, func):
        self.test, args = func(self.test)
        if args: self.args["train"] = args

        assert len(self.train) == len(self.targets)

    def __getitem__(self, i):
        #print(self.train[i], self.targets[i])
        return self.train[i], self.targets[i]

    def __len__(self):
        assert len(self.train) == len(self.targets)
        return len(self.train)

