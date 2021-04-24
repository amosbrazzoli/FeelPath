import re
from nltk.stem import SnowballStemmer

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return text

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def replace_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))

    def replace(match):
        return contraction_dict[match.group(0)]

    def replacer(text):
        return contraction_re.sub(replace, text)

    return replacer


def stemmer(textroll):
    stmmr = SnowballStemmer("english")
    return [stmmr.stem(w) for w in textroll]

def purge(textroll):
    return [ w for w in textroll if not w in {"", None, " ", ".", ","} ]

def padd_cut(max_len, padd_token="[PAD]"):
    def wrapped(text):
        if len(text) >= max_len:
            return text[:max_len]
        text += [padd_token] * (max_len - len(text))
        return text
    return wrapped

def out_dim_2d_conv(H_in: int,
                    W_in: int,
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