#!/usr/bin/env python
import json, sys
print "run e_dim h_dim lr l2_penalty epoch n dev_acc train_cost_mean dev_cost_mean tied_embeddings bidir".replace(" ", "\t")
for line in sys.stdin:
    if not line.startswith("STATS"):
        continue
    _stats, json_str = line.split("\t")  # sanity check only two
    stats = json.loads(json_str)    
    out = [stats[f] for f in ['run', 'e_dim', 'h_dim', 'lr', 'l2_penalty', 'epoch', 'n_egs_trained', 'dev_acc']]
    out.append(stats['train_cost']['mean'])
    out.append(stats['dev_cost']['mean'])
    out.append("TIED" if stats['tied_embeddings'] else "UNTIED")
    out.append("BIDIR" if stats['bidir'] else "UNIDIR")
    print "\t".join(map(str, out))

