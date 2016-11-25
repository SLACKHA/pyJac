#! /usr/bin/env python

"""
A simple script that acts as the 'EDITOR' for loopy code generation,
changing simple code-generation patterns that cause errors for various
OpenCL implementations (mostly Intel)
"""

import sys
import re

swaps = {r'-1\s*\*\s*(lid|gid)\(\d+\)' : r'- \1'}

def main(filename):
    #import pdb; pdb.set_trace()
    with open(filename, 'r') as file:
        lines = file.readlines()

    #do any replacements
    for swap in swaps:
        lines = [re.sub(swap, swaps[swap], line) for line in lines]

    with open(filename, 'w') as file:
        file.write(lines)


if __name__ == '__main__':
    main(sys.argv[1])
