import torch
from torchtext.data.utils import get_tokenizer

def token_idx(sentences):
    tokenizer = get_tokenizer("basic_english")
    args = {}
    lexicon = {}
    i = 1
    train = []
    max_len, min_len = 0, 0
    for sentence in sentences:
        sentence_idx = []
        for token in tokenizer(sentence):
            if (j := lexicon.get(token, None)) == None:
                lexicon[token] = i

                sentence_idx.append(i)
                i += 1
            else:
                sentence_idx.append(j)

        if (s:= len(sentence)) > max_len:
            max_len = s
        elif s < min_len:
            min_len = s
        
        train.append(sentence_idx)

    args["len"] = len(lexicon)
    args["min"] = min_len
    args["max"] = max_len
    return train, args

def padder(max_len, padder=0):
    def padding(iteriter):
        out = []
        for ite in iteriter:
            if len(ite) < max_len:
                for _ in range(max_len -len(ite)):
                    ite.append(padder)
            out.append(ite[:max_len])
        return out, None
    return padding

def save_checkpoint(save_path, model, valid_loss):
    if not save_path: return

    state_dict = {'model_state_dict' : model.state_dict(),
                    'valid_loss' : valid_loss}
    
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")

def load_checkpoint(load_path, model, device='cuda'):
    if not load_path: return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None: return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path, device):
    if load_path == None: return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

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

def padd_cut(max_len, padd_token="[PAD]"):
    def wrapped(text):
        if len(text) >= max_len:
            return text[:max_len]
        text += [padd_token] * (max_len - len(text))
        return text
    return wrapped

if __name__ == "__main__":
    data = ["the quick brown fox", "jumps over the lazy dog"]
    data, _ = token_idx(data)
    assert data == [[1, 2, 3, 4], [5, 6, 1, 7, 8]]

    padd = padder(7, 0)
    assert padd(data) == [[1, 2, 3, 4, 0, 0, 0], [5, 6, 1, 7, 8, 0, 0]]


