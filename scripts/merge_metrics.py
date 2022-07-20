#!/usr/bin/env python
# coding: utf-8
import multibind as mb
import numpy as np
import pandas as pd
import bindome as bd
import sys          
from pathlib import Path

if __name__ == '__main__':
    """
    prepare final file with all r2 results
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Precompute diffusion connectivities for knn data integration methods.')

    parser.add_argument('-o', '--output', required=True, help='output directory for counts and queries metadata file')    
    parser.add_argument('-i', '--input', required=True, help='input metrics files', nargs="*") 
    args = parser.parse_args()
    
    res = []
    for f in args.input:
        print(f)
        df = pd.read_csv(f)
        res.append(df)
        
    res = pd.concat(res).reset_index(drop=True)    
    res.to_csv(args.output, sep='\t')
