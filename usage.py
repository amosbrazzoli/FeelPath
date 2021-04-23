from commons import padd_cut, clean_text, clean_numbers, replace_contractions, stemmer, purge
from torchtext.data.utils import get_tokenizer
from models import MLP
import torch

MAX_LEN = 50
EMBEDDING_LEN = 100
NUM_CLASSES = 7
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
SAVE_PATH = "save/model.pt"

model = MLP(MAX_LEN, EMBEDDING_LEN, NUM_CLASSES)

model.load_state_dict(torch.load(SAVE_PATH))
model.eval()



def pipe(sentences: str):
    data = sentences.split(".")
    data = map(clean_numbers, data)
    print(list(data))
    data = map(clean_numbers, data)
    print(list(data))
    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have"}
    replacer = replace_contractions(contraction_dict)
    data = map(replacer, data)
    print(list(data))
    tokenizer = get_tokenizer('basic_english')
    data = map(tokenizer, data)
    print(list(data))
    data = map(stemmer, data)
    print(list(data))
    data = map(purge, data)
    print(list(data))
    data = map(padd_cut(MAX_LEN), data)
    print(list(data))
    ## Add embedding
    print(list(data))

if __name__ == "__main__":
    s = "mais la connoissance de le plus mals. que me ait fait du doleur. que ne sajez que faire. de qu'il me fait pair."
    pipe(s)


