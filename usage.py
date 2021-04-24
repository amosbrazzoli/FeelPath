from commons import padd_cut, clean_text, clean_numbers, replace_contractions, stemmer, purge
from torchtext.data.utils import get_tokenizer
from models import MLP
from torchtext.vocab import GloVe
import torch

MAX_LEN = 50
EMBEDDING_LEN = 100
NUM_CLASSES = 7
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
SAVE_PATH = "save/model.pt"





def pipe(sentences: str):
    sentences = sentences.strip(".")
    data = sentences.split(".")
    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have"}
    replacer = replace_contractions(contraction_dict)
    tokenizer = get_tokenizer('basic_english')
    cutter = padd_cut(MAX_LEN)
    embedding_glove = GloVe(name="6B", dim=EMBEDDING_LEN)
    out_data = []
    for d in data:
            d = clean_text(d)
            d = clean_numbers(d)
            d = replacer(d)
            d = tokenizer(d)
            d = stemmer(d)
            d = purge(d)
            d = cutter(d)
            d = [ embedding_glove[dprime]  for dprime in d]
            d = torch.stack(d) 
            out_data.append(d)
    return torch.stack(out_data)


def predict(model, sentences: str):
    x = pipe(sentences)
    pred = model(x)
    pred = torch.argmax(pred, dim=1).tolist()
    return pred


if __name__ == "__main__":
    s = "mais la connoissance de le plus mals. que me ait fait du doleur. que ne sajez que faire. de qu'il me fait pair."

    model = MLP(MAX_LEN, EMBEDDING_LEN, NUM_CLASSES)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    pred = predict(model, s)




