from commons import clean_text, clean_numbers, replace_contractions, stemmer, purge
from training import NUM_CLASSES
from models import MLP
import torch

MAX_LEN = 50
EMBEDDING_LEN = 100
NUM_CLASSES = 7
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
SAVE_PATH = "save/model.pt"

model = MLP(MAX_LEN, EMBEDDING_LEN, NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load(SAVE_PATH))
model.eval()



def pipe(sentences: str):
    data = sentences.split(".")
    
