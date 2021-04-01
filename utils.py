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



if __name__ == "__main__":
    data = ["the quick brown fox", "jumps over the lazy dog"]
    data, _ = token_idx(data)
    assert data == [[1, 2, 3, 4], [5, 6, 1, 7, 8]]

    padd = padder(7, 0)
    assert padd(data) == [[1, 2, 3, 4, 0, 0, 0], [5, 6, 1, 7, 8, 0, 0]]


