#!/usr/bin/env python

import argparse
from Bio import SeqIO
import json
import os

import sys
sys.path.insert(0, 'modules/MSAgen/')
import MSAgen

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse command line arguments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser(description = "Run Grid Search",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")
scriptArgs.add_argument("--output",
                        dest = "genoutput", 
                        metavar = "FILENAME", 
                        type=argparse.FileType("w"),
                        required=True,
                        help="Filename of the output fasta files for simulated genomes, JSON dict is named accordingly")
scriptArgs.add_argument("--N",
                        dest = "N",
                        metavar = "INT",
                        type = int,
                        default = 8,
                        help = "Number of genomes to simulate")
scriptArgs.add_argument("--seqlen",
                        dest = "seqlen",
                        metavar = "INT",
                        type = int,
                        default = 110000,
                        help = "Length of each simulated genome")
scriptArgs.add_argument("--genelen",
                        dest = "genelen",
                        metavar = "INT",
                        type = int,
                        default = 140,
                        help = "Length of simulated gene in each genome")
scriptArgs.add_argument("--coding-dist",
                        dest = "codingdist",
                        metavar = "FLOAT",
                        type = float,
                        default = 0.35,
                        help = "Evolutionary distance of simulated genes")
scriptArgs.add_argument("--noncoding-dist",
                        dest = "noncodingdist",
                        metavar = "FLOAT",
                        type = float,
                        default = 0.7,
                        help = "Evolutionary distance of simulated non-gene sequences")
args = parser.parse_args()

assert args.N > 0, "[ERROR] >>> --N must be greater than zero"
assert args.seqlen > 0, "[ERROR] >>> --seqlen must be positive"
assert args.genelen >= 0, "[ERROR] >>> --genelen must be positive or zero"

# ~~~~~~~~~~~~~~~~~~~~~~~~
# simulate or load genomes
# ~~~~~~~~~~~~~~~~~~~~~~~~

sequences, posDict = MSAgen.generate_sequences(args.N, args.seqlen, args.genelen, 
                                               coding_dist=args.codingdist, noncoding_dist=args.noncodingdist)
posDict['generatingParameters'] = {'N': args.N,
                                   'seqlen': args.seqlen, 
                                   'genelen': args.genelen, 
                                   'codingdist': args.codingdist, 
                                   'noncodingdist': args.noncodingdist}
base, ext = os.path.splitext(args.genoutput.name)
faname = base+".fa" if ext == '' else base+ext
jsonname = base+".json"
SeqIO.write(sequences, faname, "fasta")
with open(jsonname, 'wt') as fh:
    json.dump(posDict, fh, indent=2)

print("[INFO] >>> Generated simulated genomes and stored to", args.genoutput.name, "with accompanying json")
