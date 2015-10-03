# snli hacking

hacking with the Stanford Natural Language Inference corpus http://nlp.stanford.edu/projects/snli/

# tldr results

( all for 100 dev )

model | # train | dev accuracy
----- | ----- | --------
log_reg_baseline.py | 100 | 0.35
log_reg_baseline.py | 1K | 0.45
log_reg_baseline.py | 10K | 0.56
log_reg_baseline.py | 500K+ (all) | 0.67
nn_baseline.py | 100 | 0.39
nn_baseline.py | 1K | 0.47
nn_baseline.py | 10K | 0.58
nn_baseline.py | 500K_ (all) | 0.63

# simple baseline

features...

* all tokens in sentence 1 prepended with "s1_"
* all tokens in sentence 2 prepended with "s2_"

## 100 train, 100 dev

```
$ ./log_reg_baseline.py --num-from-train=100 --num-from-dev=100
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
$ ./log_reg_baseline.py --num-from-train=1000 --num-from-dev=100
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
$ ./log_reg_baseline.py --num-from-train=10000 --num-from-dev=100
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
$ time ./log_reg_baseline.py --num-from-train=-1 --num-from-dev=100

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

# nn_baseline

```
$ ./nn_baseline.py --embedding-dim=20 --hidden-dim=20 --learning-rate=0.01 --num-from-train=100 --num-from-dev=100 --num-epochs=10



# vocab check

```
time cat data/snli_1.0_train.jsonl \
 | ./parse_distinct_tokens.py \
 | sort -k2 -nr \
 > token_freq.tsv
```

36_391 entries (nice and small!)

but an unusual set compared to, say, the 1e6 sentence corpus...

token | 1e6 freq | token | snli freq
----- | -------- | ----- | ---------
. | 970059 | a | 1441039
of | 845203 | . | 964030
and | 645219 | the | 535493
in | 602064 | in | 407662
to | 488035 | is | 374068
a | 482654 | man | 266490
is | 243535 | on | 236203
' | 241019 | and | 206529
was | 239712 | are | 199381
-lrb- | 237887 | of | 192415
-rrb- | 237400 | with | 169513
` | 212268 | woman | 137794
as | 197400 | two | 122247
for | 185954 | people | 121335
by | 162542 | , | 114538
with | 162149 | to | 114072
on | 160348 | at | 98790
that | 150584 | wearing | 81141
's | 148546 | an | 80334
'' | 124864 | his | 72550
`` | 122000 | young | 61596
from | 110950 | men | 61112
his | 109739 | playing | 59568
he | 109146 | girl | 59345
it | 108952 | boy | 58354
at | 100304 | white | 57115
are | 93788 | shirt | 56578
an | 87625 | while | 56323
were | 85952 | black | 55133
which | 83635 | dog | 54026
