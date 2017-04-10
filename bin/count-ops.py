#!/usr/bin/env python
import re

longest = 0
with open("optimal3.txt", "r") as f:
    for line in f:
        m = re.match(r'.*? ([^(]*)', line)
        if not m:
            continue
        line = m.group(1)
        #line = f.readline()
        #line = line.strip()
        c = re.split(r'\s+', line)
        if len(c) > longest:
            print c
            longest = len(c)
        #print line
