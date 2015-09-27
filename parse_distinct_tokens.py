#!/usr/bin/env python
from collections import Counter
import json
import sys


def tokens_in_parse(parse):
    for token in parse.split(" "):
        if token != "(" and token != ")":
            yield token.lower()


freqs = Counter()
for line in sys.stdin:
    d = json.loads(line)
    for token in tokens_in_parse(d['sentence1_binary_parse']):
        freqs[token] += 1
    for token in tokens_in_parse(d['sentence2_binary_parse']):
        freqs[token] += 1
for token, freq in freqs.iteritems():
    print "%s\t%s" % (token, freq)
