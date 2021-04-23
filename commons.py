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