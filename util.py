from collections import Counter
import json
import numpy as np
import theano

def tokens_in_parse(parse):
    for token in parse.split(" "):
        if token != "(" and token != ")":
            yield token.lower()

def tokens_in_sentences(eg):
    tokens_in_s1 = list(tokens_in_parse(eg['sentence1_binary_parse']))
    tokens_in_s2 = list(tokens_in_parse(eg['sentence2_binary_parse']))
    return (tokens_in_s1, tokens_in_s2)

def label_for(eg):
    labels = ['contradiction', 'neutral', 'entailment']
    try:
        return labels.index(eg['gold_label'])
    except ValueError:
        return None

def load_data(dataset, vocab, max_egs=None, update_vocab=True):
    stats = Counter()
    x, y = [], []
    for line in open(dataset, "r"):
        eg = json.loads(line)
        l = label_for(eg)
        if l is None:
            stats['n_ignored'] += 1
        else:
            s1, s2 = tokens_in_sentences(eg)
            s1 = vocab.ids_for_tokens(s1, update_vocab)
            stats['n_tokens'] += len(s1)
            stats['n_unk'] += len(s1) - len(filter(None, s1))
            s2 = vocab.ids_for_tokens(s2, update_vocab)
            stats['n_tokens'] += len(s2)
            stats['n_unk'] += len(s2) - len(filter(None, s2))
            x.append((s1, s2))
            y.append(l)
        if len(x) == max_egs:
            break
    return x, y, stats

def sharedMatrix(n_rows, n_cols, name, orthogonal_init=True):
    if orthogonal_init and n_rows < n_cols:
        print >>sys.stderr, "warning: can't do orthogonal init of %s, since n_rows (%s) < n_cols (%s)" % (name, n_rows, n_cols)
        orthogonal_init = False
    w = np.random.randn(n_rows, n_cols)
    if orthogonal_init:
        w, _s, _v = np.linalg.svd(w, full_matrices=False)
    return theano.shared(np.asarray(w, dtype='float32'), name=name, borrow=True)

def accuracy(confusion):
    return np.sum(confusion * np.identity(3)) / np.sum(confusion)


