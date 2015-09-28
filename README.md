# snli hacking

hacking with the Stanford Natural Language Inference corpus http://nlp.stanford.edu/projects/snli/

# vocab

time cat data/snli_1.0_train.jsonl \
 | ./parse_distinct_tokens.py \
 | sort -k2 -nr \
 > token_freq.tsv

36_391 entries (nice and small!)

# simple baseline

features...

* all tokens in sentence 1 prepended with "s1_"
* all tokens in sentence 2 prepended with "s2_"

## 100 train, 100 dev

```
$ ./log_reg_baseline.py --num-from-train=100
train_ignored 0 dev_ignored 1
train confusion
[[32  0  0]
 [ 0 35  0]
 [ 0  0 33]]
dev confusion (accuracy)
[[ 7  8 22]
 [ 6  9 16]
 [ 7  6 19]] (0.35)
```

## 1_000 train, 100 dev

```
$ ./log_reg_baseline.py --num-from-train=1000
train_ignored 2 dev_ignored 1
train confusion
[[312  10  11]
 [  9 304  20]
 [ 16  13 305]]
dev confusion (accuracy)
[[20  7 10]
 [ 7 12 12]
 [10  9 13]] (0.45)
```

## 10_000 train, 100 dev

```
./log_reg_baseline.py
train_ignored 12 dev_ignored 1
train confusion
[[2671  277  386]
 [ 305 2590  431]
 [ 298  262 2780]]
dev confusion (accuracy)
[[19  7 11]
 [ 6 16  9]
 [10  1 21]] (0.56)
```

## all train, 100 dev

```
$ time ./log_reg_baseline.py

|train| 549367 train_ignored 785 dev_ignored 1
train confusion
[[121313  27799  34075]
 [ 29942 117733  35089]
 [ 23658  20902 138856]]
dev confusion (accuracy)
[[21  6 10]
 [ 4 23  4]
 [ 7  2 23]] (0.67)

(6m 6s)
```

