#!/usr/bin/env python
from collections import Counter
import json
import sys
from tokenise_parse import *

freqs = Counter()
for line in sys.stdin:
    d = json.loads(line)
    for token in tokens_for(d, 1, 'PARSE_WITH_OPEN_CLOSE_TAGS'):
        freqs[token] += 1
    for token in tokens_for(d, 2, 'PARSE_WITH_OPEN_CLOSE_TAGS'):
        freqs[token] += 1
for token, freq in freqs.iteritems():
    print "%s\t%s" % (token, freq)
