#!/usr/bin/env python

# read a vocab and precompute a .npy embedding matrix. for each vocab entry that's in
# the provided 300d glove embeddings use the glove data. if it's not, generate a random
# vector but scale it to the median length of the glove embeddings. "reserve" idx 0
# in the matrix for UNK embedding.
import argparse
import numpy as np
import sys
from sklearn import random_projection

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, help="reference vocab of non glove data; token \t idx")
parser.add_argument("--glove-data", required=True, help="glove data. ssv, token, e_d1, e_d2, ...")
parser.add_argument("--npy", required=True, help="npy output")
parser.add_argument("--random-projection-dimensionality", default=None, type=float, 
                    help="if set we randomly project the glove data to a smaller dimensionality")
opts = parser.parse_args()

# slurp vocab entries. assume idxs are valid, ie 1 < i < |v|, no dups, no gaps, etc
# (recall reserving 0 for UNK)
vocab = {}  # token => idx
for line in open(opts.vocab, "r"):
    token, idx = line.strip().split("\t")
    assert idx != "0", "expecting to reserve 0 for UNK"
    vocab[token] = int(idx)
print "vocab has", len(vocab), "entries"

# alloc output after we see first glove embedding (so we know it's length)
embeddings = None
glove_dimensionality = None

# pass over glove data copying data into embedddings array
# for the cases where the token is in the reference vocab.
tokens_requiring_random = set(vocab.keys())
glove_embedding_norms = []
for line in open(opts.glove_data, "r"):
    cols = line.strip().split(" ")
    token = cols[0]
    if token in vocab:
        glove_embedding = np.array(cols[1:], dtype=np.float32)
        if embeddings is None:
            glove_dimensionality = len(glove_embedding)
            embeddings = np.empty((len(vocab)+1, glove_dimensionality), dtype=np.float32)  # +1 for unk
        assert len(glove_embedding) == glove_dimensionality, "differing dimensionality in glove data?"
        embeddings[vocab[token]] = glove_embedding
        tokens_requiring_random.remove(token)
        glove_embedding_norms.append(np.linalg.norm(glove_embedding))

# given these embeddings we can calculate the median norm of the glove data
median_glove_embedding_norm = np.median(glove_embedding_norms)

print >>sys.stderr, "after passing over glove there are", len(tokens_requiring_random), \
    "tokens requiring a random alloc"

# return a random embedding with the same norm as the glove data median norm
def random_embedding():
    random_embedding = np.random.randn(1, glove_dimensionality)
    random_embedding /= np.linalg.norm(random_embedding)
    random_embedding *= median_glove_embedding_norm
    return random_embedding

embeddings[0] = random_embedding()  # UNK
for token in tokens_requiring_random:
    embeddings[vocab[token]] = random_embedding()

# randomly project (if configured to do so)
if opts.random_projection_dimensionality is not None:
    p = random_projection.GaussianRandomProjection(n_components=opts.random_projection_dimensionality)
    embeddings = p.fit_transform(embeddings)

# write embeddings npy to disk
np.save(opts.npy, embeddings)



