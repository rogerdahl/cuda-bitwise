#!/usr/bin/env python

import re

def main():
    print len(get_r_missing())
    r_missing = get_r_missing()
    b_missing = get_b_missing()
    print len(r_missing)
    #print len(b_missing)
    #b_dup()

def get_r_missing():
    found_dict = {}
    with open('bitwise-15.txt', 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'(.{16}).*', line)
            if not m:
                print 'No match: {}'.format(line)
                exit()
            found_dict[m.group(1)] = line

    missing_list = []
    for i in range(2**16):
        b = '{:016b}'.format(i)
        if b not in found_dict:
            missing_list.append(b)

    return missing_list

def get_b_missing():
    found_dict = {}
    with open('optimal3.txt', 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'([01]{16})(.*)', line)
            if not m:
                print 'No match: {}'.format(line)
                continue
            if len(m.group(2)):
                found_dict[m.group(1)] = line

    missing_list = []
    for i in range(2**16):
        b = '{:016b}'.format(i)
        if b not in found_dict:
            missing_list.append(b)

    return missing_list

if __name__ == '__main__':
    main()
