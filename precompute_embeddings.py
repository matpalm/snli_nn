#!/usr/bin/env python

# read a vocab and precompute a .npy embedding matrix. for each vocab entry that's in 
# the provided 300d glove embeddings use the glove data. if it's not, generate a random
# vector but scale it to the median length of the glove embeddings. 
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", required=True, help="input vocab; token => idx")
parser.add_argument("--glove-data", required=True, help="input vocab; token => idx")
parser.add_argument("--npy", required=True, help="npy output")
opts = parser.parse_args()

# slurp vocab entries. assume idxs are valid, ie 0 < i < |v|, no dups, no gaps, etc
vocab = {}  # token => idx
for line in open(opts.vocab, "r"):
    token, idx = line.strip().split("\t")
    vocab[token] = int(idx)
print "vocab has", len(vocab), "entries"

# assign output
embeddings = np.empty((len(vocab), 300), dtype=np.float32)

# pass over glove data copying embeddings across
tokens_requiring_random = set(vocab.keys())
glove_embedding_norms = []
for line in open(opts.glove_data, "r"):
    cols = line.strip().split(" ")
    token = cols[0]
    if token in vocab:
        glove_embedding = np.array(cols[1:], dtype=np.float32)
        embeddings[vocab[token]] = glove_embedding
        tokens_requiring_random.remove(token)
        glove_embedding_norms.append(np.lingalg.norm(glove_embedding))        
print >>sys.stderr, "after passing over glove there are", len(tokens_requiring_random), \
    "tokens requiring a random alloc"

for token in tokens_requiring_random:
    
    
    
# pass again over file building up embeddings
for row, line in enumerate(open(opts.input, "r")):
    cols = line.strip().split(" ")
    token = cols.pop(0)
    embeddings[row] = np.array(cols, dtype=np.float32)
    vocab_out.write("%s\t%s\n" % (token, row))
    
np.save(opts.npy, embeddings)
vocab_out.close()


