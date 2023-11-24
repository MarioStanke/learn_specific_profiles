#!/usr/bin/env python3

import argparse
from Bio import SeqIO
import logging
import numpy as np
import os
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from modules import prepare, generate, parseAndCompose, mplog



def get_output_dir():
    """ Create output directory name from config and return it. 
        Expample: `out_1000Exons_species_20_15_20_50_15_20_15_20_mammals` """
    lengths_config_str = str(args.min_left_exon_len)
    lengths_config_str += "_" + str(args.left_exon_anchor_len)
    lengths_config_str += "_" + str(args.min_left_intron_len)
    lengths_config_str += "_" + str(args.min_exon_len)
    lengths_config_str += "_" + str(args.middle_exon_anchor_len)
    lengths_config_str += "_" + str(args.min_right_intron_len)
    lengths_config_str += "_" + str(args.right_exon_anchor_len)
    lengths_config_str += "_" + str(args.min_right_exon_len)

    def getFilename(path):
        return os.path.splitext(os.path.split(path)[1])[0]

    # dirs
    output_dir = os.path.join(args.path, 
                              f"out_{'' if not args.n else str(args.n) + 'Exons_'}{getFilename(args.species)}" \
                                  + f"_{lengths_config_str}_{getFilename(args.hal)}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def make_stats_table():
    """ Create result overview from the created exon files """
    df = pd.DataFrame(columns = ["path", "exon", "exon_len", "human_seq_len",
                                 "exon_len_to_human_len_ratio", "median_len",
                                 "exon_len_to_median_len_ratio","average_len",
                                 "exon_len_to_average_len", "num_seqs", "ambiguous"])
    dir = get_output_dir() if args.hal else args.stats_table
    for exon in os.listdir(dir):
        exon_dir = os.path.join(dir, exon)
        if os.path.isdir(exon_dir):
            try:
                exon_coords = list(map(int, exon.split("_")[2:]))
            except ValueError:
                logging.error(f"Could not parse exon coordinates from {exon}, skipping in stats table!")
                continue

            exon_len = exon_coords[1] - exon_coords[0]
            lens = []
            for record in SeqIO.parse(os.path.join(exon_dir, "combined.fasta"), "fasta"):
                lens.append(len(record.seq))
                if record.id.startswith("Homo_sapiens"):
                    human_len = len(record.seq)

            median_len =  np.median(lens)
            average_len = np.average(lens)

            if os.path.exists(os.path.join(exon_dir, "true_alignment_exon_contains_ambiguous_bases.clw")):
                ambiguous = 1
            elif os.path.exists(os.path.join(exon_dir, "true_alignment.clw")):
                ambiguous = -1
            else:
                ambiguous = 0

            new_row_dict = {"path": exon_dir,
                            "exon": exon,
                            "exon_len": exon_len,
                            "human_seq_len": human_len,
                            "exon_len_to_human_len_ratio": exon_len/human_len,
                            "median_len": median_len,
                            "exon_len_to_median_len_ratio": exon_len/median_len,
                            "average_len": average_len,
                            "exon_len_to_average_len": exon_len/average_len,
                            "num_seqs": len(lens),
                            "ambiguous": ambiguous}

            df.loc[len(df)] = new_row_dict

    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    df.to_csv(os.path.join(dir, 'stats_table.csv'), index=True, header=True, line_terminator='\n', sep=";")
    return df


# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creating a dataset of exons from hg38.bed and .hal alignment. Only \
                                                  exons which are between introns')
    parser.add_argument('--hg38', help = 'path to hg38-refseq.bed')
    parser.add_argument('--hal', help = 'path to the .hal file')
    parser.add_argument('--species', help = 'path to species file, that contains a species name per line. ' \
                    + 'Species names must match those in `--hal`. These species are the target for liftover from human')
    parser.add_argument('--min_left_exon_len', type = int, default = 20, 
                        help = 'Left neighbouring exon must be at least this long')
    parser.add_argument('--min_left_intron_len', type = int, default = 20, 
                        help = 'Left neighbouring intron must be at least this long')
    parser.add_argument('--min_right_exon_len', type = int, default = 20, 
                        help = 'Right neighbouring exon must be at least this long')
    parser.add_argument('--min_right_intron_len', type = int, default = 20, 
                        help = 'Right neighbouring intron must be at least this long')
    parser.add_argument('--min_exon_len', type = int, default = 50, help = 'Middle exon must be at least this long')
    parser.add_argument('--middle_exon_anchor_len', type = int, default = 15, 
                        help = 'the middle of the exon is also lifted, to check whether it is between left and right \
                                exons if target .bed')
    parser.add_argument('--left_exon_anchor_len', type = int, default = 15, 
                        help = 'Length of the fragment from the left exon that is used for halLiftover')
    parser.add_argument('--right_exon_anchor_len', type = int, default = 15, 
                        help = 'Length of the fragment from the right exon that is used for halLiftover')
    parser.add_argument('--path', default = ".", help = 'working directory')
    parser.add_argument('-n', type = int, help = 'limit the number of exons to n')
    parser.add_argument('--num-processes', type = int, default = 1, help = 'number of processes to use', 
                        dest='num_processes')
    parser.add_argument('-v', action = 'store_true', help = 'verbose')
    parser.add_argument('--use_old_bed', action = 'store_true', 
                        help = 'use the old bed files and dont calculate new ones')
    parser.add_argument('--use_old_fasta', action = 'store_true', 
                        help = 'use the old fasta files and dont calculate new ones')
    parser.add_argument('--discard_multiple_bed_hits', action = 'store_true', 
                        help = 'sometimes, halLiftover maps a single coordinate to 2 or more, if this flag is passed, \
                                the species is discarded, otherwise the largest of the hits is selected')
    parser.add_argument('--stats_table', nargs = '?', const = True, 
                        help ='instead of getting all the exon data, get stats table of existing data. Specified path, \
                               or pass hg38, hal and species and same n')
    parser.add_argument('--convert_short_lc_to_UC', type = int, default = 0, 
                        help = 'Also store a combined.fasta where softmasekd parts at most as long as this are ' \
                               + 'converted to upper case')
    parser.add_argument('--logfile', help = 'Path to logfile. If not specified, log to stdout', 
                        type = argparse.FileType('w'))
    parser.add_argument('--loglevel', default = 'INFO', choices = ['DEBUG', 'INFO', 'WARNING'],
                        help = 'Set the logging level. Must be one of DEBUG, INFO, WARNING')
    args = parser.parse_args()

    assert args.left_exon_anchor_len < args.min_left_exon_len, "[ERROR] >>> left_exon_anchor_len > min_left_exon_len"
    assert args.right_exon_anchor_len < args.min_right_exon_len, \
        "[ERROR] >>> right_exon_anchor_len > min_right_exon_len"
    assert args.middle_exon_anchor_len < args.min_exon_len, "[ERROR] >>> middle_exon_anchor_len > min_exon_len"
    assert args.loglevel in ['DEBUG', 'INFO', 'WARNING'], "[ERROR] >>> loglevel must be one of DEBUG, INFO, WARNING"
    # TODO im using the above also for the start and end of th middle exon, not only the middle of the middle/current exon

    # set logging
    logformat = '[%(asctime)s] %(levelname)s: %(message)s'
    if args.loglevel == 'DEBUG':
        loglevel = logging.DEBUG
    elif args.loglevel == 'INFO':
        loglevel = logging.INFO
    elif args.loglevel == 'WARNING':
        loglevel = logging.WARNING

    if args.logfile:
        #logging.basicConfig(filename=args.logfile.name, encoding='utf-8', level=loglevel, format=logformat)
        mplog.setup_logger(log_file = args.logfile.name, level = loglevel)
    else:
        #logging.basicConfig(level=loglevel, format=logformat)
        mplog.setup_logger(level = loglevel)

    # start the workflow
    if not args.stats_table:
        assert args.hg38 and args.hal and args.species, "[ERROR] >>> You must pass path to hg38, hal and species.lst"
        output_dir = get_output_dir()

        # files
        json_path = os.path.join(output_dir, "filtered_internal_exons.json")

        overwrite = False
        if os.path.exists(json_path):
            print("There exists a file with previously exported filtered_internal_exons with same config")
            print("do you want to overwrite it? [y/n] ", end = "")
            while True:
                x = input().strip()
                if x == "y":
                    overwrite = True
                    break
                elif x == "n":
                    overwrite = False
                    break
                else:
                    print("your answer must be either y or n")

        hg38_refseq_bed = parseAndCompose.parse_bed(args.hg38) # load_hg38_refseq_bed()

        # ADDITIONAL FILTERING: only use NM_... transcripts (and only those with at least 3 exons)
        hg38_refseq_bed = hg38_refseq_bed[hg38_refseq_bed['name'].map(lambda x: x.startswith("NM_")) \
                                          & hg38_refseq_bed['blockCount'].map(lambda x: x >= 3)]

        filtered_internal_exons = prepare.get_to_be_lifted_exons(hg38_refseq_bed, json_path, overwrite, args)
        logging.info(f"Got {len(filtered_internal_exons)} exons to be lifted")
        logging.debug("Set of chromosomes in filtered_internal_exons: " + str(sorted(set([exon.seq for exon in filtered_internal_exons]))))
        generate.create_exon_data_sets(filtered_internal_exons, output_dir, args)
        make_stats_table()
    else:
        if not os.path.isdir(args.stats_table):
            assert args.hg38 and args.hal and args.species, \
                "you must pass path to hg38, hal and species.lst or path to the dir of which the stats table should \
                    get created"
        make_stats_table()
