# snli hacking

hacking with the Stanford Natural Language Inference corpus http://nlp.stanford.edu/projects/snli/

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

# vocab check

```
time cat data/snli_1.0_train.jsonl \
 | ./parse_distinct_tokens.py \
 | sort -k2 -nr \
 > token_freq.tsv
```

36_391 entries (nice and small!)

but an unusual set ....

```
head -n30 token_freq.tsv 
a    1441039
.    964030
the  535493
in   407662
is   374068
man  266490
on   236203
and  206529
are  199381
of   192415
with 169513
woman	137794
two	122247
people	121335
,	114538
to	114072
at	98790
wearing	81141
an	80334
his	72550
young	61596
men	61112
playing	59568
girl	59345
boy	58354
white	57115
shirt	56578
while	56323
black	55133
dog	54026
```

compared to 1e6 sentences dataset..

```
head -n30 token_freq.tsv
.    970059
of   845203
and  645219
in   602064
to   488035
a    482654
is   243535
'    241019
was  239712
-lrb-	237887
-rrb-	237400
`	212268
as	197400
for	185954
by	162542
with	162149
on	160348
that	150584
's	148546
''	124864
``	122000
from	110950
his	109739
he	109146
it	108952
at	100304
are	93788
an	87625
were	85952
which	83635
```