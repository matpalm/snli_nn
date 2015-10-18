#!/usr/bin/env python

# read a vocab and precompute a .npy embedding matrix. for each vocab entry that's in
# the provided 300d glove embeddings use the glove data. if it's not, generate a random
# vector but scale it to the median length of the glove embeddings.
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, help="reference vocab; token => idx")
parser.add_argument("--glove-data", required=True, help="glove data. ssv, token, e_d1, e_d2, ...")
parser.add_argument("--npy", required=True, help="npy output")
opts = parser.parse_args()

# slurp vocab entries. assume idxs are valid, ie 0 < i < |v|, no dups, no gaps, etc
vocab = {}  # token => idx
for line in open(opts.vocab, "r"):
    token, idx = line.strip().split("\t")
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
            embeddings = np.empty((len(vocab), glove_dimensionality), dtype=np.float32)
        assert len(glove_embedding) == glove_dimensionality, "differing dimensionality in glove data?"
        embeddings[vocab[token]] = glove_embedding
        tokens_requiring_random.remove(token)
        glove_embedding_norms.append(np.linalg.norm(glove_embedding))

median_glove_embedding_norm = np.median(glove_embedding_norms)

print >>sys.stderr, "after passing over glove there are", len(tokens_requiring_random), \
    "tokens requiring a random alloc"
for token in tokens_requiring_random:
    random_embedding = np.random.randn(1, glove_dimensionality)
    random_embedding /= np.linalg.norm(random_embedding)
    random_embedding *= median_glove_embedding_norm
    embeddings[vocab[token]] = random_embedding

# write embeddings npy to disk
np.save(opts.npy, embeddings)



