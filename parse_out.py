#!/usr/bin/env python
import json, sys

fields = set()
all_stats = []
for line in sys.stdin:
    if not line.startswith("STATS"):
        continue        
    _stats, json_str = line.split("\t")  # sanity check only two
    stats = json.loads(json_str)    
    all_stats.append(stats)
    fields.update(stats.keys())
fields = list(fields)

print "\t".join(fields)

for stats in all_stats:
    values = []
    for field in fields:
        value = stats[field] if field in stats else "NA"
        if field == 'train_cost':
            value = stats['train_cost']['mean']
        elif field == 'dev_cost':
            value = stats['dev_cost']['mean']
        elif field == 'tied_embeddings':
            value = "TIED" if value==True else "UNTIED"
        elif field == 'bidir':
            value = "BIDIR" if value==True else "UNIDIR"
        elif field == 'swap_symmetric_examples':
            value = "SWAP" if value==True else "NO_SWAP"
        values.append(value)
    print "\t".join(map(str, values))

