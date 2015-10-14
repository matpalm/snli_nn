#!/usr/bin/env python
import json, sys
fields = None
for line in sys.stdin:
    if not line.startswith("STATS"):
        continue        
    _stats, json_str = line.split("\t")  # sanity check only two
    stats = json.loads(json_str)    
    if 'update_fn' not in stats:
        stats['update_fn'] = 'vanilla'
    if fields is None:
        fields = stats.keys()
        print "\t".join(fields)
    values = []
    for field in fields:
        value = stats[field]
        if field == 'train_cost':
            value = stats['train_cost']['mean']
        elif field == 'dev_cost':
            value = stats['dev_cost']['mean']
        elif field == 'tied_embeddings':
            value = "TIED" if value else "UNTIED"
        elif field == 'bidir':
            value = "BIDIR" if value else "UNIDIR"
        values.append(value)
    print "\t".join(map(str, values))

