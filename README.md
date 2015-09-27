
# vocab

time cat data/snli_1.0_train.jsonl \
 | ./parse_distinct_tokens.py \
 | sort -k2 -nr \
 > token_freq.tsv

36_391 entries (nice and small!)

# TODOS

- parse out tokens vs pos tags
- work out how to build a sequence representation of binary parses
- dual sequences later? pos tags / dparse / tokens in parallel sequences
- build vocab
- run baseline: "s1:" & "s2:" prepended to tokens then NB / log reg wataever

