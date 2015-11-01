#!/usr/bin/env python
from collections import Counter
import json
import numpy as np
import tokenise_parse
import sys

PARSE_MODE = sys.argv[1]

def quantiles(v):
    return np.percentile(v, np.linspace(0, 100, 5))

token_freq = Counter()
s1_lengths = []
s2_lengths = []
for line in sys.stdin:
    eg = json.loads(line)
    s1_tokens = tokenise_parse.tokens_for(eg, 1, PARSE_MODE)
    s1_lengths.append(len(s1_tokens))
    s2_tokens = tokenise_parse.tokens_for(eg, 2, PARSE_MODE)
    s2_lengths.append(len(s2_tokens))
    token_freq.update(s1_tokens)
    token_freq.update(s2_tokens)

print "s1_lengths quantiles", quantiles(s1_lengths)
print "s2_lengths quantiles", quantiles(s2_lengths)
print token_freq.most_common(30)

