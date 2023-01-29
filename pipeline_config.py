from pathlib import Path
import snakemake.io
from collections import defaultdict
import itertools
from os.path import join


class ParsedConfig:
    def __init__(self, config, gene_names='gene_names.txt'):

        # TODO: define and check schema of config

        self.ROOT = Path(config["ROOT"]).resolve()
        self.EXPERIMENT = config["EXPERIMENT"]
        self.py_env = config["py_env"]
        self.ANNOTATIONS = config['ANNOTATIONS']
        self.HYPERPARAM = config['HYPERPARAM']
        # print(config['GENE_NAMES'])
        self.GENE_NAMES = [s.strip() for s in open(gene_names)] # takes all gene names
        # self.GENE_NAMES = config['GENE_NAMES']
        # print(self.GENE_NAMES)
        # assert False


    def get_all_file_patterns(self, file_type):
        return join(self.ROOT, f"{file_type}.csv")

