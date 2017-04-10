#!/usr/bin/env python

import sys

#program = ['v1', 'v2', 'not']
program = sys.argv[1:]
stack = []

for op in program:
    if op == 'v1':
        stack.append(0b1010101010101010)
    elif op == 'v2':
        stack.append(0b1100110011001100)
    elif op == 'v3':
        stack.append(0b1111000011110000)
    elif op == 'v4':
        stack.append(0b1111111100000000)
    elif op == 'and':
        stack.append(stack.pop() & stack.pop())
    elif op == 'or':
        stack.append(stack.pop() | stack.pop())
    elif op == 'eor':
        stack.append(stack.pop() ^ stack.pop())
    elif op == 'not':
        stack.append(~stack.pop())
    else:
        print 'Ignoring invalid op: {}'.format(op)

for v in stack:
    print '{:016b}'.format(v & 0xffff)
