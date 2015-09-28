#!/usr/bin/env python
import argparse
import json
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="data/snli_1.0_train.jsonl")
parser.add_argument("--num-from-train", default=None)
parser.add_argument("--dev", default="data/snli_1.0_dev.jsonl")
#parser.add_argument("--test", default="data/snli_1.0_test.jsonl")
args = parser.parse_args()

print args
exit

def tokens_in_parse(parse):
    for token in parse.split(" "):
        if token != "(" and token != ")":
            yield token.lower()


def features_for(eg):
    features = []
    for token in tokens_in_parse(eg['sentence1_binary_parse']):
        features.append("s1_" + token)
    for token in tokens_in_parse(eg['sentence2_binary_parse']):
        features.append("s2_" + token)
    return features


def label_for(eg):
    try:
        return {'contradiction': 0,
                'neutral': 1,
                'entailment': 2}[eg['gold_label']]
    except KeyError:
        return None


def load_data(dataset, max_egs=None):
    x, y, n_ignored = [], [], 0
    for line in open(dataset, "r"):
        eg = json.loads(line)
        l = label_for(eg)
        if l is None:
            n_ignored += 1
        else:
            x.append(" ".join(features_for(eg)))
            y.append(l)
        if len(x) == max_egs:
            break
    return x, y, n_ignored


# train model
train_x, train_y, train_ignored = load_data(args.train, 
                                            max_egs=int(args.num_from_train))
x_vectorizer = CountVectorizer(binary=True)
train_x_v = x_vectorizer.fit_transform(train_x)
model = linear_model.LogisticRegression()
model.fit(train_x_v, train_y)

# sanity check model against data it was trained on
train_pred = model.predict(train_x_v)

# try against dev data
dev_x, dev_y, dev_ignored = load_data(args.dev, max_egs=100)
dev_x_v = x_vectorizer.transform(dev_x)
dev_pred = model.predict(dev_x_v)

print "|train|", len(train_x),
print "train_ignored", train_ignored,
print "dev_ignored", dev_ignored

def accuracy(confusion):
    return np.sum(confusion * np.identity(3)) / np.sum(confusion)

train_c = confusion_matrix(train_y, train_pred)
print "train confusion\n", train_c, accuracy(train_c)

dev_c = confusion_matrix(dev_y, dev_pred)
print "dev confusion (accuracy)\n", dev_c, "(%s)" % accuracy(dev_c)
