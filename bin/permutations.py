#!/usr/bin/env python

import itertools

p = list(itertools.permutations([1, 2, 3, 4], 4))

print len(p)

for x in p:
    a, b, c, d = x
    if a >= b or c >= d:
        continue
    print x

