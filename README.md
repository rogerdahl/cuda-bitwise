# CUDA Bitwise

Finding the minimal sequence of boolean operations that yields a given truth table.

The program generates a table of the minimal sequence of boolean operations needed in a stack based language in order implement a given truth table for up to 4 boolean inputs.

The table can also be used for synthesising the minimal number of logical ports in hardware.

A copy of the complete table is included, in `minimal-bitwise-result.txt`.

Operations:

* `vX`: Push the boolean value of input X onto the stack
* `and`, `or`, `eor`: Pop the two top values off the stack, perform the bitwise operation on them, and push the result back on the stack
* `not`: Pop the top value off the stack, invert it, and push the result back on the stack

## Example

This is a random entry from the generated table:

    0001001100011111: v3 v1 v2 or and v2 v4 and or not

The binary sequence describes this truth table:


| v4 v3 v2 v1 | output |
---|---
0000 | 0
0001 | 0
0010 | 0
0011 | 1
0100 | 0
0101 | 0
0110 | 1
0111 | 1
1000 | 0
1001 | 0
1010 | 0
1011 | 1
1100 | 1
1101 | 1
1110 | 1
1111 | 1

When the entire sequence has been performed, in this case, `v3 v1 v2 or and v2 v4 and or not`, a single binary value will be left, corresponding to the binary input.

E.g., `v1=1`, `v2=0`, `v3=1`, `v4=0`, would yield the value `0`.

## Building on Linux

```
$ git clone <copy/paste the clone link from the green button on top of this page>
$ cd cuda-bitwise
$ mkdir cmake-build-release
$ cd cmake-build-release
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
