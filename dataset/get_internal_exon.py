#!/usr/bin/env python3

import argparse
#import Bio
from Bio import Align, AlignIO, Seq, SeqIO, SeqRecord
#from Bio.Align import MultipleSeqAlignment
#from Bio.SeqRecord import SeqRecord
from collections import namedtuple
import json
import math
import numpy as np
import os
import pandas as pd
import re
import shutil
import subprocess
import time



########################################################################################################################
# define named tuple for exon to make it safer to use in a proper editor/IDE
Exon = namedtuple("Exon", ["seq", "start_in_genome", "stop_in_genome", "id", 
                           "left_anchor_start", "left_anchor_end", "right_anchor_start", "right_anchor_end",
                           "left_intron_len", "right_intron_len", "strand", "substring_len", "total_len"])

# for the same reason, create simple class for liftover_seq
class LiftoverSeq:
    def __init__(self):
        self.seq_start_in_genome = None
        self.seq_stop_in_genome = None
        self.middle_of_exon_start = None
        self.middle_of_exon_stop = None
        self.on_reverse_strand = None
        self.seq_name = None
        self.substring_len = None

# same reason, must match the named tuple returned by Pandas itertuples() method, check this later
BedRow = namedtuple("BedRow", ["Index", "chrom", "chromStart", "chromEnd", "name", "score", "strand", 
                               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"])
BedColumns = BedRow._fields[1:] # chrom, chromStart, chromEnd, ...



########################################################################################################################
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



########################################################################################################################
def parse_bed(bedfile):
    """ Load a bed file into a Pandas DataFrame and return it. """
    bedf = pd.read_csv(bedfile, delimiter = "\t", header = None)
    assert bedf.columns.size <= 12, "[ERROR] >>> bed file has more than 12 columns"
    assert bedf.columns.size >= 3, "[ERROR] >>> bed file has less than 3 columns"
    ncol = bedf.columns.size
    bedf.columns = BedColumns[:ncol]

    def parseBlockList(s):
        fields = s.split(",")
        if fields[-1] == '':
            return [int(a) for a in fields[:-1]]
        else:
            return [int(a) for a in fields]

    if ncol >= 11:
        bedf["blockSizes"] = bedf["blockSizes"].map(parseBlockList)
        assert all(bedf["blockCount"] == bedf["blockSizes"].map(len)), "[ERROR] >>> blockCount != len(blockSizes)"
    if ncol >= 12:
        bedf["blockStarts"] = bedf["blockStarts"].map(parseBlockList)
        assert all(bedf["blockCount"] == bedf["blockStarts"].map(len)), "[ERROR] >>> blockCount != len(blockStarts)"

    # just to be safe
    def ensureNumericColumn(column):
        if column in bedf.columns:
            if not pd.api.types.is_numeric_dtype(bedf[column]):
                bedf[column] = pd.to_numeric(bedf[column])

        return bedf
    
    bedf = ensureNumericColumn("chromStart")
    bedf = ensureNumericColumn("chromEnd")
    bedf = ensureNumericColumn("score")
    bedf = ensureNumericColumn("thickStart")
    bedf = ensureNumericColumn("thickEnd")
    bedf = ensureNumericColumn("blockCount")

    return bedf



########################################################################################################################
def load_hg38_refseq_bed():
    """ Load hg38 refseq bed file (specified in arguments) as Pandas DataFrame and return it. """
    start = time.perf_counter()
    print("[INFO] >>> started load_hg38_refseq_bed()")
    hg38_refseq_bed = pd.read_csv(args.hg38, delimiter = "\t", header = None)
    assert hg38_refseq_bed.columns.size == 12, \
        f"[ERROR] >>> hg38 refseq bed has {hg38_refseq_bed.columns.size} columns, 12 expected"
    hg38_refseq_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", 
                               "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    def parseBlockList(s):
        fields = s.split(",")
        if fields[-1] == '':
            return [int(a) for a in fields[:-1]]
        else:
            return [int(a) for a in fields]

    hg38_refseq_bed["blockSizes"] = hg38_refseq_bed["blockSizes"].map(parseBlockList)
    hg38_refseq_bed["blockStarts"] = hg38_refseq_bed["blockStarts"].map(parseBlockList)
    assert all(hg38_refseq_bed["blockCount"] == hg38_refseq_bed["blockSizes"].map(len)), \
        "[ERROR] >>> blockCount != len(blockSizes)"
    assert all(hg38_refseq_bed["blockCount"] == hg38_refseq_bed["blockStarts"].map(len)), \
        "[ERROR] >>> blockCount != len(blockStarts)"

    # just to be safe
    def ensureNumericColumn(column):
        if not pd.api.types.is_numeric_dtype(hg38_refseq_bed[column]):
            hg38_refseq_bed[column] = pd.to_numeric(hg38_refseq_bed[column])

        return hg38_refseq_bed
    
    hg38_refseq_bed = ensureNumericColumn("chromStart")
    hg38_refseq_bed = ensureNumericColumn("chromEnd")
    hg38_refseq_bed = ensureNumericColumn("score")
    hg38_refseq_bed = ensureNumericColumn("thickStart")
    hg38_refseq_bed = ensureNumericColumn("thickEnd")
    hg38_refseq_bed = ensureNumericColumn("blockCount")

    print("[INFO] >>> finished load_hg38_refseq_bed(). It took:", time.perf_counter() - start)
    return hg38_refseq_bed



########################################################################################################################
def get_all_internal_exons(hg38_refseq_bed: pd.DataFrame):
    """ Get all internal exons from the given hg38 refseq bed file and return them as a dictionary where the keys are
        (chromosome, exon start, exon stop) and the values are lists of rows that mapped to this exon range. """
    start = time.perf_counter()
    print("[INFO] >>> started get_all_internal_exons()")
    internal_exons = {} # keys: (chromosome, exon start, exon stop), values: list of rows that mapped to this exon range
    # row fields are ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", 
    #                 "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    for row in hg38_refseq_bed.itertuples():
        # assert that row matches the BedRow namedtuple, this allows to hint the type of the row as BedRow
        assert row._fields == BedRow._fields, "[ERROR] >>> BedRow namedtuple fields do not match hg38_refseq_bed " \
            + f"columns: {BedRow._fields} != {row._fields}"
        row = BedRow(*row) # this is entirely for the IDE to know which fields to expect, remove if performance impacted

        if row.blockCount < 3: # need at least 3 exons to have an internal exon
            continue

        assert row.chromStart <= row.chromEnd, \
            f'[ERROR] >>> chromStart ({row.chromStart}) > chromEnd ({row.chromEnd}) in hg38 refseq bed file'
        assert row.thickStart <= row.thickEnd, \
            f'[ERROR] >>> thickStart ({row.thickStart}) > thickEnd ({row.thickEnd}) in hg38 refseq bed file'

        # excluding first and last exon, yields list [(exon_len, exon_start), (exon_len, exon_start), ...]
        chromosome = row.chrom
        for exon_len, exon_start in zip(row.blockSizes[1:-1], row.blockStarts[1:-1]):
            exon_start_in_genome = row.chromStart + exon_start
            exon_end_in_genome = row.chromStart + exon_start + exon_len # the end pos is not included
            key = (chromosome, exon_start_in_genome, exon_end_in_genome)
            if key not in internal_exons:
                internal_exons[key] = []

            internal_exons[key].append(row)

    if args.v:
        print("[INFO|V] >>> Since the same exon occurs in multiple genes, which may just be spliced differently") # wat?
        for key in sorted(internal_exons.keys()):
            if len(internal_exons[key]) > 3:
                print(f"  key is exon = {key}")
                print("  value is list of df_rows:")
                for df_row in internal_exons[key]:
                    print("   ", df_row)
                break

    print("[INFO] >>> finished get_all_internal_exons(). It took:", time.perf_counter() - start)
    return internal_exons



########################################################################################################################
def filter_and_choose_exon_neighbours(all_internal_exons: dict[tuple, list[BedRow]]):
    """ Because of alternatively spliced genes, an exon might have multiple neighboring exons to choose from. 
        Currently, simply the first splice form that matches the requirements is chosen for each exon. """
    start = time.perf_counter()
    print("[INFO] >>> started filter_and_choose_exon_neighbours()")
    filtered_internal_exons = []
    for key in all_internal_exons.keys():
        chromosome, exon_start_in_genome, exon_stop_in_genome = key
        if args.n and len(filtered_internal_exons) >= args.n:
            break

        # filter out exons that do not match requirements given in arguments (neighbor exon lengths etc.)
        for row in all_internal_exons[key]:
            # key: (chromosome, exon start in genome, exon end in genome)
            exon_start_in_gene = exon_start_in_genome - row.chromStart
            exon_id = row.blockStarts.index(exon_start_in_gene) # dies if exon_start_in_gene not found in blockStarts
            assert row.blockSizes[exon_id] == exon_stop_in_genome - exon_start_in_genome, \
                f"[ERROR] >>> blockSize {row.blockSizes[exon_id]} of exon {exon_id} is not same as calculated stop - " \
                    + f"start ({exon_stop_in_genome} - {exon_start_in_genome}) in genome (exon: {key}, row: {row})"
            if exon_id == 0 or exon_id == len(row.blockSizes) - 1:
                print(f"[WARNING] >>> exon_id == {exon_id} for exon {key} in row {row}")
                continue
            if row.blockSizes[exon_id] < args.min_exon_len:
                continue
            if row.blockSizes[exon_id - 1] < args.min_left_exon_len:
                continue
            left_intron_len = row.blockStarts[exon_id] - row.blockStarts[exon_id-1] - row.blockSizes[exon_id - 1]
            if left_intron_len < args.min_left_intron_len:
                continue
            right_intron_len = row.blockStarts[exon_id + 1] - row.blockStarts[exon_id] - row.blockSizes[exon_id]
            if right_intron_len < args.min_right_intron_len:
                continue
            if row.blockSizes[exon_id + 1] < args.min_right_exon_len:
                continue

            # getting coordinates from left/rightmost end of left/right exon that will be lifted to the other genomes
            left_anchor_start = exon_start_in_genome - left_intron_len - args.left_exon_anchor_len
            left_anchor_end = left_anchor_start + args.left_exon_anchor_len
            right_anchor_start = exon_stop_in_genome + right_intron_len
            right_anchor_end = right_anchor_start + args.right_exon_anchor_len

            e = Exon(seq = chromosome,
                     start_in_genome = exon_start_in_genome,
                     stop_in_genome = exon_stop_in_genome,
                     id = exon_id,
                     left_anchor_start = left_anchor_start,
                     left_anchor_end = left_anchor_end,
                     right_anchor_start = right_anchor_start,
                     right_anchor_end = right_anchor_end,
                     left_intron_len = left_intron_len,
                     right_intron_len = right_intron_len,
                     strand = row.strand,
                     substring_len = right_anchor_start - left_anchor_end,
                     total_len = right_anchor_end - left_anchor_start)
            filtered_internal_exons.append(e)

            # for an internal exon the neigbhours are choosen (simply the first occurance is selected)
            # so other occurances of this exon in alternatively spliced genes (= rows in hg38.bed) are discarded
            break

    if args.v:
        for i in range(3):
            print("[INFO|V] >>> filtered_internal_exons[", i, "]")
            for key in sorted(filtered_internal_exons[i].keys()):
                print("  ", key, filtered_internal_exons[i][key])
    print("[INFO] >>> Finished filter_and_choose_exon_neighbours(). It took:", time.perf_counter() - start)
    return filtered_internal_exons



########################################################################################################################
def write_filtered_internal_exons(filtered_internal_exons: list[Exon], json_path):
    """ Store the filtered internal exons in a json file for later use. Avoid recomputing them. """
    start = time.perf_counter()
    print("[INFO] >>> Started write_filtered_internal_exons()")
    # convert Exon objects to dicts for json.dump()
    exon_list = [e._asdict() for e in filtered_internal_exons]
    with open(json_path, "w") as json_out:
        json.dump(exon_list, json_out, indent=2)
    print("[INFO] >>> Finished write_filtered_internal_exons(). It took:", time.perf_counter() - start)



########################################################################################################################
def get_to_be_lifted_exons(hg38_refseq_bed, json_path, overwrite):
    """ Either load the filtered internal exons from a json file or compute them. """
    if os.path.exists(json_path) and not overwrite:
        print(f"[INFO] >>> The file {json_path} exists, so it isn't computed again")
        start = time.perf_counter()
        print(f"[INFO] >>> Started json.load({json_path})")
        with open(json_path) as file:
            filtered_internal_exons = json.load(file)
        print(f"[INFO] >>> Finished json.load({json_path}). It took:", time.perf_counter() - start)
    else:
        all_internal_exons = get_all_internal_exons(hg38_refseq_bed)
        filtered_internal_exons = filter_and_choose_exon_neighbours(all_internal_exons)
        write_filtered_internal_exons(filtered_internal_exons, json_path)
    if args.n and len(filtered_internal_exons) > args.n:
        filtered_internal_exons = filtered_internal_exons[:args.n]

    return filtered_internal_exons



########################################################################################################################
def create_human_liftover_bed(exon: Exon, out_path):
    """ Create and write to disk a bed formatted file for a single human exon, 
        containing three lines with the left, right and middle lift areas, respectively. """
    # seq     start           stop            name    score   strand
    # chr1    67093589        67093604        left    0       -
    with open(out_path, "wt") as bed_file:
        def add_bed_line(start, stop, name, seq = exon.seq, score = "0", strand = exon.strand):
            bed_file.write("\t".join([str(seq), str(start), str(stop), str(name), str(score), str(strand)]) + "\n")
            
        # name will match "\d+_\d+_\d+_(left|right|middle)"
        base_name = f"exon_{exon.start_in_genome}_{exon.stop_in_genome}_{exon.id}"

        # left and right neighbouring exon
        add_bed_line(start = str(exon.left_anchor_start), 
                     stop = str(exon.left_anchor_end), 
                     name = f"{base_name}_left")
        add_bed_line(start = str(exon.right_anchor_start), 
                     stop = str(exon.right_anchor_end), 
                     name = f"{base_name}_right")

        # middle exon
        left_middle = (exon.stop_in_genome + exon.start_in_genome - args.middle_exon_anchor_len)//2
        right_middle = left_middle + args.middle_exon_anchor_len #this pos is not part of the area to be lifted
        add_bed_line(start = str(left_middle),
                     stop = str(right_middle),
                     name = f"{base_name}_middle")

        # start and stop of exon

        # add_bed_line(start = str(exon["start_in_genome"]),
        #              stop = str(exon["start_in_genome"] + args.middle_exon_anchor_len),
        #              name = f"{base_name}_exonstart")
        # add_bed_line(start = str(exon["stop_in_genome"] -  args.middle_exon_anchor_len),
        #              stop = str(exon["stop_in_genome"]),
        #              name = f"{base_name}_exonend")



########################################################################################################################
def liftover(human_exon_to_be_lifted_path, species_name, out_dir, exon: Exon): # DEBUG: exon argument
    """ Either creates new bed file by calling halLiftover with the human exon bed file and the species name,
        or searches for an old  (only if --use_old_bed is given). 
        Returns named tuple with returncode that can be checked for success (returncode 0) """
    bed_file_path = os.path.join(out_dir, species_name+".bed")
    if not args.use_old_bed:
        command = f"time halLiftover {args.hal} Homo_sapiens {human_exon_to_be_lifted_path} {species_name} " \
                    + f"{bed_file_path}"
        print("[INFO] >>> running:", command)
        status = subprocess.run(command, shell=True, capture_output=True)
        #os.system(command)

        # ~~~ DEBUG ~~~
        if species_name == "Homo_sapiens":
            die = False
            print("[DEBUG] >>> Homo sapiens self liftover")
            print("[DEBUG] >>> Exon:", exon)
            # create fasta from exon, human_exon_bed and lifted bed
            exonfasta = os.path.join(out_dir, "DEBUG_human_exon.fasta")
            bedfasta = os.path.join(out_dir, "DEBUG_human_bed.fasta")
            liftfasta = os.path.join(out_dir, "DEBUG_human_lift.fasta")
            run_hal_2_fasta(species_name, exon.left_anchor_start, exon.total_len, exon.seq, exonfasta)
            
            hgbed = parse_bed(human_exon_to_be_lifted_path)
            assert len(hgbed.index) == 3, f"[ERROR] >>> human exon bedfile has {len(hgbed.index)} lines (3 expected)"
            assert hgbed.iloc[0]['name'][-4:] == 'left', "[ERROR] >>> human exon bedfile not as expected:\n"+str(hgbed)
            assert hgbed.iloc[1]['name'][-5:] == 'right', "[ERROR] >>> human exon bedfile not as expected:\n"+str(hgbed)
            bedstart = hgbed.iloc[0]['chromStart']
            bedend = hgbed.iloc[1]['chromEnd']
            assert bedstart < bedend, "[ERROR] >>> left start after right end:\n"+str(hgbed)
            bedlen = bedend-bedstart
            run_hal_2_fasta(species_name, bedstart, bedlen, exon.seq, bedfasta)

            liftbed = parse_bed(bed_file_path)
            assert len(liftbed.index) == 3, f"[ERROR] >>> lifted bedfile has {len(liftbed.index)} lines (3 expected)"
            assert liftbed.iloc[0]['name'][-4:] == 'left', "[ERROR] >>> lifted bedfile not as expected:\n"+str(liftbed)
            assert liftbed.iloc[1]['name'][-5:] == 'right', "[ERROR] >>> lifted bedfile not as expected:\n"+str(liftbed)
            bedstart = liftbed.iloc[0]['chromStart']
            bedend = liftbed.iloc[1]['chromEnd']
            assert bedstart < bedend, "[ERROR] >>> left start after right end:\n"+str(liftbed)
            bedlen = bedend-bedstart
            run_hal_2_fasta(species_name, bedstart, bedlen, exon.seq, liftfasta)

            # load generated fastas and compare (should all be the same, 
            #   especially lifted over sequence from human to human)
            exonseq = []
            with open(exonfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    exonseq.append(record)

            print(f"[DEBUG] >>> number of exon sequences: {len(exonseq)}")
            print("[DEBUG] >>> Exon sequence:")
            print(exonseq[0])

            bedseq = []
            with open(bedfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    bedseq.append(record)

            print(f"[DEBUG] >>> number of bed sequences: {len(bedseq)}")
            print("[DEBUG] >>> BED sequence:")
            print(bedseq[0])

            liftseq = []
            with open(liftfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    liftseq.append(record)

            print(f"[DEBUG] >>> number of lifted sequences: {len(liftseq)}")
            print("[DEBUG] >>> Lifted sequence:")
            print(liftseq[0])

            def printStackedSeqs(seq1: str, seq2: str, lw=20):
                a1 = a2 = 0
                b1 = b2 = 0
                while True:
                    b1 = min(len(seq1), b1+lw)
                    b2 = min(len(seq2), b2+lw)
                    s1 = seq1[a1:b1]
                    s2 = seq2[a2:b2]
                    
                    if len(s1) + len(s2) == 0:
                        break

                    m = ''.join(['|' if s1[i] == s2[i] else '-' for i in range(min(len(s1),len(s2)))])
                    print(s1)
                    print(m)
                    print(s2)
                    print()

                    a1 = b1
                    a2 = b2



            if exonseq[0].seq != bedseq[0].seq:
                print("[WARNING] >>> Exon seq and BED seq differ")
                printStackedSeqs(exonseq[0].seq, bedseq[0].seq, 120)
                die = True
            else:
                print("[DEBUG] >>> Exon seq and BED seq are the same")

            if exonseq[0].seq != liftseq[0].seq:
                print("[WARNING] >>> Exon seq and lifted seq differ")
                printStackedSeqs(exonseq[0].seq, liftseq[0].seq, 120)
                die = True
            else:
                print("[DEBUG] >>> Exon seq and lifted seq are the same")

            if bedseq[0].seq != liftseq[0].seq:
                print("[WARNING] >>> BED seq and lifted seq differ")
                printStackedSeqs(bedseq[0].seq, liftseq[0].seq, 120)
                die = True
            else:
                print("[DEBUG] >>> BED seq and lifted seq are the same")

            #print("[DEBUG] >>> Lifted BED:")
            #print(liftbed)
            if die:
                print("[ERROR] >>> Exiting due to errors")
                exit(1)
            #exit(0)

        # ~~~ /DEBUG ~~~

        return status
    else:
        mock_status = namedtuple("mock_status", ["returncode", "message"]) # simulate subprocess status with returncode
        bed_files = [f for f in os.listdir(out_dir) if f.endswith(".bed")]
        for bed_file in bed_files:
            # if bed_file.startswith(single_species):
            #     return f"{out_dir}/{bed_file}"
            if bed_file == f"{species_name}.bed":
                return mock_status(0, f"Found {os.path.join(out_dir, bed_file)}")
            
        return mock_status(1, f"Found no file that matched {os.path.join(out_dir, species_name+'.bed')}")
    


########################################################################################################################
def extract_info_and_check_bed_file(bed_dir, species_name, exon: Exon, liftover_seq: LiftoverSeq):
    """ Checks the generated bedfile from halLiftover and extracts sequence information into liftover_seq.
        Sequence coordinates exclude the left and right anchors.
        Returns true if everything was successful, false if something was wrong with the bed file. """
    bed_file_path = os.path.join(bed_dir, species_name+".bed")
    if os.path.getsize(bed_file_path) == 0:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_empty.bed"))
        return False

    species_bed = pd.read_csv(bed_file_path, delimiter = "\t", header = None)
    assert species_bed.columns.size == 6, \
        f"[ERROR] >>> Bed file {bed_file_path} has {species_bed.columns.size} columns, 6 expected"
    species_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]

    if len(species_bed.index) != 3 and args.discard_multiple_bed_hits:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_more_than_3_lines.bed"))
        return False
    
    l_m_r = {} # find longest bed hit for each left, middle and right exon
    for row in species_bed.itertuples():
        x = re.search(r"\d+_\d+_\d+_(.+)", row.name)
        try:
            # `:=`: assign x.group(1) to y if it exists, then y = left, right or middle
            if (y := x.group(1)) in l_m_r:
                # TODO there is exactly one line for each left, middle and right, so this should be unnecessary
                len_of_previous_bed_hit = l_m_r[y]["chromEnd"] - l_m_r[y]["chromStart"]
                len_of_current_bed_hit = row.chromEnd - row.chromStart
                if len_of_current_bed_hit > len_of_previous_bed_hit:
                    l_m_r[y] = row._asdict()
            else:
                l_m_r[y] = row._asdict()
        # catch any error and print error message
        except Exception as e:
            print("[ERROR] >>> An error occured:", e)
            print("[ERROR] >>> l_m_r[x.group(1)] didnt work")
            print("            row.name", row.name)
            print("            bed_file_path", bed_file_path)
            exit(1)
        # Exception might miss some errors, so catch all errors
        except:
            print("[ERROR] >>> l_m_r[x.group(1)] didnt work")
            print("            row.name", row.name)
            print("            bed_file_path", bed_file_path)
            exit(1)

    if len(l_m_r) != 3:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_not_all_l_m_r.bed"))
        return False
    if l_m_r["left"]["strand"] != l_m_r["right"]["strand"] or l_m_r["left"]["strand"] != l_m_r["middle"]["strand"]:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_unequal_strands.bed"))
        return False
    if l_m_r["left"]["chrom"] != l_m_r["left"]["chrom"] or l_m_r["left"]["chrom"] != l_m_r["middle"]["chrom"]:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_unequal_seqs.bed"))
        return False

    # anchors get cut away here!
    if exon.strand == l_m_r["left"]["strand"]:
        liftover_seq.seq_start_in_genome = l_m_r["left"]["chromEnd"]
        liftover_seq.seq_stop_in_genome = l_m_r["right"]["chromStart"]
    else:
        # if strand is opposite to human, left and right swap
        # I think        [left_start,   left_stop] ... [middle_start, middle_stop] ... [right_start, right_stop]
        # gets mapped to [right_start, right_stop] ... [middle_start, middle_stop] ... [left_start,   left_stop]
        liftover_seq.seq_start_in_genome  = l_m_r["right"]["chromEnd"]
        liftover_seq.seq_stop_in_genome  = l_m_r["left"]["chromStart"]

    liftover_seq.middle_of_exon_start = l_m_r["middle"]["chromStart"]
    liftover_seq.middle_of_exon_stop = l_m_r["middle"]["chromEnd"]

    if liftover_seq.seq_start_in_genome >= liftover_seq.middle_of_exon_start:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_left_greater_middle.bed"))
        return False
    if liftover_seq.middle_of_exon_stop >= liftover_seq.seq_stop_in_genome:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_right_less_middle.bed"))
        return False

    liftover_seq.on_reverse_strand = (l_m_r["left"]["strand"] == "-")
    liftover_seq.seq_name = l_m_r['left']['chrom']
    liftover_seq.substring_len = liftover_seq.seq_stop_in_genome - liftover_seq.seq_start_in_genome

    threshold = 1
    l1 = liftover_seq.substring_len
    l2 = exon.substring_len
    if abs(math.log10(l1) - math.log10(l2)) >= threshold:
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_lengths_differ_substantially.bed"))
        return False

    return True



########################################################################################################################
def write_extra_data_to_fasta_description_and_reverse_complement(fa_path, liftover_seq: LiftoverSeq, exon: Exon):
    """ Reads single sequence fasta from fa_path, adds exon description as a json string to the sequence,
        converts the sequence to reverse complement if the exon is on the - strand, writes the sequence back to 
        fa_path """

    ### ----------------------------------------------------------------------------------------------------------------
    ### TODO: [ME] at this point I probably wanna review and extend the information that is stored in the fasta header
    ###             to include the position information that I need for profile finding evaluation
    ### ----------------------------------------------------------------------------------------------------------------

    record = None
    for i, entry in enumerate(SeqIO.parse(fa_path, "fasta")):
        record = entry
        assert i == 0, f"[ERROR] >>> found more than one seq in fasta file {fa_path}"
        assert len(record.seq) == liftover_seq.seq_stop_in_genome - liftover_seq.seq_start_in_genome, \
            "[ERROR] >>> non stripped: actual seq len and calculated coordinate len differ"

    if record is None:
        print(f"[WARNING] >>> no seq found in fasta file {fa_path}, skipping")
        return

    # write coordinates in genome to seq description
    with open(fa_path, "wt") as out_file:
        # Extracetd fasta is from + strand
        # if exon is on - strand, TAA -> gets converted to TTA
        # these coords are from bed file and hg38:
        # 1 start in genome +     2 exon start +      3 exon stop +      4 stop in genome +
        # these are adjusted to the reversed and complemented fasta
        # 4 start in genomce cd

        description = {"seq_start_in_genome_+_strand": liftover_seq.seq_start_in_genome,
                       "seq_stop_in_genome_+_strand": liftover_seq.seq_stop_in_genome,
                       "exon_start_in_human_genome_+_strand": exon.start_in_genome,
                       "exon_stop_in_human_genome_+_strand": exon.stop_in_genome,
                       "seq_start_in_genome_cd_strand": liftover_seq.seq_start_in_genome \
                            if not liftover_seq.on_reverse_strand else liftover_seq.seq_stop_in_genome,
                       "seq_stop_in_genome_cd_strand": liftover_seq.seq_start_in_genome \
                            if liftover_seq.on_reverse_strand else liftover_seq.seq_stop_in_genome,
                       "exon_start_in_human_genome_cd_strand": exon.start_in_genome \
                            if not liftover_seq.on_reverse_strand else exon.stop_in_genome,
                       "exon_stop_in_human_genome_cd_strand": exon.stop_in_genome \
                            if not liftover_seq.on_reverse_strand else exon.start_in_genome}
        record.description = json.dumps(description)
        if liftover_seq.on_reverse_strand:
            reverse_seq = record.seq.reverse_complement()
            record.seq = reverse_seq
        SeqIO.write(record, out_file, "fasta")



########################################################################################################################
def run_hal_2_fasta(species_name, start, len, seq, outpath):
    """ Executes shell command hal2fasta that extracts a subsequence from the genome of a species in the hal-file and 
        prints the first 10 lines of the output fasta file """
    command = f"time hal2fasta {args.hal} {species_name} --start {start} --length {len} --sequence {seq} \
                --ucscSequenceNames --outFaPath {outpath}"
    print("[INFO] >>> Running:", command)
    status = subprocess.run(command, shell=True, capture_output=True)
    if not status.returncode == 0:
        print(f"[WARNING] >>> hal2fasta failed with error code {status.returncode}")
        print(status.stderr.decode("utf-8"))
    else:
        #os.system(command)
        print(f"[INFO] >>> Running: head {outpath}")
        status = subprocess.run(f"head -n 5 {outpath}", shell=True, capture_output=True)
        print(status.stdout.decode("utf-8"))
        #os.system(f"head {outpath}")



########################################################################################################################
# TODO: can I maybe use the old fasta description such that i dont have to pass liftover_seq
def strip_seqs(fasta_file, exon: Exon, out_path, liftover_seq: LiftoverSeq):
    """ If fasta_file exists, strips the sequence in the fasta file and writes the stripped sequence to `out_path`.
        It will cut `min_left_exon_len/2` and `min_right_exon_len/2` bases from each side of the
        sequence.
        `out_path` will be overwritten when fasta_file contains more than one sequence and only the last stripped 
        sequence will be stored in `out_path` """
    if os.path.exists(fasta_file):
        for record in SeqIO.parse(fasta_file, "fasta"):
            # TODO if there is more than one record, out_path will be overwritten every time
            with open(out_path, "wt") as stripped_seq_file:
                left_strip_len = int(args.min_left_exon_len/2)
                right_strip_len = int(args.min_right_exon_len/2)
                record.seq = record.seq[left_strip_len:-right_strip_len]
                seq_start_in_genome = liftover_seq.seq_start_in_genome + left_strip_len
                seq_stop_in_genome = seq_start_in_genome + liftover_seq.substring_len - left_strip_len - right_strip_len
                assert seq_stop_in_genome - seq_start_in_genome == len(record.seq), \
                    "[ERROR] >>> Stripped: actual seq len and calculated coordinate len differ"
                description = {"seq_start_in_genome_+_strand": seq_start_in_genome,
                               "seq_stop_in_genome_+_strand": seq_stop_in_genome,
                               "exon_start_in_human_genome_+_strand": exon.start_in_genome,
                               "exon_stop_in_human_genome_+_strand": exon.stop_in_genome,
                               "seq_start_in_genome_cd_strand": seq_start_in_genome \
                                   if not liftover_seq.on_reverse_strand else seq_stop_in_genome,
                               "seq_stop_in_genome_cd_strand": seq_start_in_genome \
                                   if liftover_seq.on_reverse_strand else seq_stop_in_genome,
                               "exon_start_in_human_genome_cd_strand": exon.start_in_genome \
                                   if not liftover_seq.on_reverse_strand else exon.stop_in_genome,
                               "exon_stop_in_human_genome_cd_strand": exon.stop_in_genome \
                                   if not liftover_seq.on_reverse_strand else exon.start_in_genome}
                record.description = json.dumps(description)
                SeqIO.write(record, stripped_seq_file, "fasta")



########################################################################################################################
def convert_short_lc_to_UC(outpath, input_files, threshold):
    """ Converts short (up to threshold length) softmasked parts of the sequences in input_files to upper case and
        append the results to outpath. The sequences in input_files are assumed to be in fasta format. """
    def capitalize_lowercase_subseqs(seq):
        # (?<![a-z]) negative lookbehind, (?![a-z]) negative lookahead, i.e. only match if the character before and 
        #   after the match is not a lowercase letter
        pattern = f"(?<![a-z])([a-z]{{1,{threshold}}})(?![a-z])" 
        def repl(match):
            return match.group(1).upper()
        result = re.sub(pattern, repl, seq)
        return str(result)
    
    # TODO this looks horrible, fix it and use SeqIO
    with open(outpath, "wt") as output_handle:
        for input_file in input_files:
            with open(outpath, "a") as output_handle, open(input_file, "r") as in_file_handle:
                for i, record in enumerate(SeqIO.parse(in_file_handle, "fasta")):
                    assert i == 0, f"convert_short_lc_to_UC found more than one seq in fasta file {input_file}"
                    new_seq = capitalize_lowercase_subseqs(str(record.seq))
                    output_handle.write(f">{record.id} {record.description}\n")
                    output_handle.write(f"{new_seq}\n")
                    # new_record = record.__class__(seq = "ACGT", id="homo", name="name", description="description")
                    # SeqIO.write(new_record, output_handle, "fasta")
                    # this produced
                    # File "/usr/lib/python3/dist-packages/Bio/File.py", line 72, in as_handle
                    # with open(handleish, mode, **kwargs) as fp:


########################################################################################################################
def get_input_files_with_human_at_0(from_path):
    """ Gets all file paths of .fa file from from_path, returns them as a list with the Homo_sapiens fasta as first
        element. """
    input_files = [os.path.join(from_path, f) for f in os.listdir(from_path) if f.endswith(".fa")]
    input_files = sorted(input_files, key = lambda x: 0 if re.search("Homo_sapiens", x) else 1)
    #assert re.search("Homo_sapiens", input_files[0]), f"[ERROR] >>> Homo sapiens not in first pos of {from_path}"
    if not re.search("Homo_sapiens", input_files[0]):
        print(f"[WARNING] >>> Homo sapiens not in first pos of {from_path}")

    return input_files



########################################################################################################################
def combine_fasta_files(output_file, input_files):
    """ Write all sequences from the fasta files in input_files to output_file """
    # TODO this looks weird, also out.close() should be unneccessary, refactor this
    with open(output_file, "wt") as out:
        for input_file in input_files:
            for seq_record in SeqIO.parse(input_file, "fasta"):
                SeqIO.write(seq_record, out, "fasta")
    out.close()



########################################################################################################################
def write_clw_true_state_seq(fasta_path, out_dir_path):
    """ Write clw format file with true state sequence of the human sequence as MSA to `out_dir_path`. 
        Example:
        ```
        coords_fasta                        in fasta 2050, in genome 101641763
        numerate_line                       |         |         |         |         |
        Homo_sapiens.chr1                   ttttctattttccttctagAATGCTGGTAGCTGTGTAATTAGGCAAGTTC
        true_seq                            lllllllllllllllllllEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        ```
    """
    # make sure the fasta contains a human sequence
    try:
        fasta_data = SeqIO.parse(fasta_path, "fasta")
        for record in fasta_data:
            if re.search("Homo_sapiens", record.id):
                human_fasta = record
                # if nothing is found this will call except block
        try:
            human_fasta.id
        except:
            print("[ERROR] >>> no human id found in", fasta_path)
            return
    except:
        print("[ERROR] >>> SeqIO could not parse", fasta_path)
        return

    try:
        coords = json.loads(re.search("({.*})", human_fasta.description).group(1))
    except:
        print("[ERROR] >>> Could not extract coordinates from fasta header", human_fasta.description)
        exit(1) # die

    on_reverse_strand = coords["exon_start_in_human_genome_cd_strand"] != coords["exon_start_in_human_genome_+_strand"]
    # create state sequence: llll..llllEEEE..EEEErrrr..rrrr
    if not on_reverse_strand:
        true_seq = "l" * (coords["exon_start_in_human_genome_+_strand"] - coords["seq_start_in_genome_+_strand"])
        true_seq += "E" * (coords["exon_stop_in_human_genome_+_strand"] - coords["exon_start_in_human_genome_+_strand"])
        true_seq += "r" * (coords["seq_stop_in_genome_+_strand"] - coords["exon_stop_in_human_genome_+_strand"])
    else:
        true_seq = "l" * (coords["seq_start_in_genome_cd_strand"] - coords["exon_start_in_human_genome_cd_strand"])
        true_seq += "E" * (coords["exon_start_in_human_genome_cd_strand"] 
                           - coords["exon_stop_in_human_genome_cd_strand"])
        true_seq += "r" * (coords["exon_stop_in_human_genome_cd_strand"] - coords["seq_stop_in_genome_cd_strand"])

    true_seq_record = SeqRecord.SeqRecord(seq = Seq.Seq(true_seq), id = "true_seq")

    # create sequence of bars every 10 bases: |         |         |         |         | ...
    len_of_line_in_clw = 50
    numerate_line = ""
    for i in range(len(human_fasta.seq)):
        i_line = i % len_of_line_in_clw
        if i_line % 10 == 0:
            numerate_line += "|"
        else:
            numerate_line += " "

    numerate_line_record =  SeqRecord.SeqRecord(seq = Seq.Seq(numerate_line), id = "numerate_line")

    # create descriptive sequence for each clw-Block (50 bases), giving the current position in the fasta and genome
    coords_fasta = ""
    for line_id in range(len(human_fasta.seq)//len_of_line_in_clw):
        in_fasta = line_id*len_of_line_in_clw
        if not on_reverse_strand:
            coords_line = f"in fasta {in_fasta}, in genome {in_fasta + coords['seq_start_in_genome_+_strand']}"
        else:
            coords_line = f"in fasta {in_fasta}, in genome {coords['seq_start_in_genome_cd_strand']- in_fasta}"
        coords_fasta += coords_line + " " * (len_of_line_in_clw - len(coords_line))

    last_line_len = len(human_fasta.seq) - len(coords_fasta)
    coords_fasta += " " * last_line_len
    coords_fasta_record = SeqRecord.SeqRecord(seq = Seq.Seq(coords_fasta), id = "coords_fasta")

    # create "MSA" of the records
    records = [coords_fasta_record, numerate_line_record, human_fasta, true_seq_record]
    alignment = Align.MultipleSeqAlignment(records)

    filesuffix = "" # if exon sequence contains lowercase or N, add this to the filename
    for base, e_or_i in zip(human_fasta.seq, true_seq_record.seq):
        if e_or_i == "E" and base in "acgtnN":
            filesuffix = "_exon_contains_lc_or_N"
            break

    alignment_out_path = f"{out_dir_path}/true_alignment{filesuffix}.clw"
    with open(alignment_out_path, "w") as output_handle:
        AlignIO.write(alignment, output_handle, "clustal")
    print("wrote alignment to", alignment_out_path)



########################################################################################################################
def create_exon_data_sets(filtered_internal_exons: list[Exon], output_dir):
    """ For each exon in filtered_internal_exons, computes liftover for each species and stores all sequences in a
        fasta file """
    with open(args.species, "rt") as species_file:
        species = species_file.readlines()

    all_species = [s.strip() for s in species]

    for exon in filtered_internal_exons:
        exon_dir = os.path.join(output_dir, f"exon_{exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}")
        bed_output_dir = os.path.join(exon_dir, "species_bed")
        seqs_dir = os.path.join(exon_dir, "species_seqs")
        non_stripped_seqs_dir = os.path.join(seqs_dir, "non_stripped")
        stripped_seqs_dir = os.path.join(seqs_dir, "stripped")
        capitalzed_subs_seqs_dir = os.path.join(exon_dir,f"combined_fast_capitalized_{args.convert_short_lc_to_UC}")
        
        for d in [exon_dir, bed_output_dir, seqs_dir, non_stripped_seqs_dir, 
                  stripped_seqs_dir, capitalzed_subs_seqs_dir]:
            os.makedirs(d, exist_ok = True)

        # Quick fix to avoid reruns after failed jobs TODO do this properly
        output_file = os.path.join(capitalzed_subs_seqs_dir, "combined.fasta") \
            if args.convert_short_lc_to_UC > 0 else os.path.join(exon_dir, "combined.fasta")
        if os.path.isfile(output_file):
            print(f"[WARNING] >>> Skipping exon {exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}",
                  f"because {output_file} already exists")
            continue

        human_exon_bedfile = os.path.join(exon_dir, "human_exons.bed")
        create_human_liftover_bed(exon = exon, out_path = human_exon_bedfile)
        for single_species in all_species:
            bed_status = liftover(human_exon_bedfile, single_species, bed_output_dir, exon)
            if bed_status.returncode == 1:
                print(f"[WARNING] >>> No bed file for species {single_species}: {bed_status}")
                continue

            liftover_seq = LiftoverSeq()
            if not extract_info_and_check_bed_file(
                bed_dir = bed_output_dir,
                species_name = single_species,
                exon = exon,
                liftover_seq = liftover_seq
            ):
                continue


            # getting the seq, from human: [left exon [lifted]] [intron] [exon] [intron] [[lifted] right exon]
            # the corresponding seq of [intron] [exon] [intron] in other species
            out_fa_path = os.path.join(non_stripped_seqs_dir, single_species+".fa")
            if not args.use_old_fasta:
                run_hal_2_fasta(species_name = single_species,
                                start = liftover_seq.seq_start_in_genome,
                                len = liftover_seq.substring_len,
                                seq = liftover_seq.seq_name,
                                outpath = out_fa_path)

                write_extra_data_to_fasta_description_and_reverse_complement(fa_path = out_fa_path,
                                                                             liftover_seq = liftover_seq,
                                                                             exon = exon)

            stripped_fasta_file_path = re.sub("non_stripped","stripped", out_fa_path)
            strip_seqs(fasta_file = out_fa_path,
                       exon = exon,
                       out_path = stripped_fasta_file_path,
                       liftover_seq = liftover_seq)

            # create alignment of fasta and true splice sites
            if single_species == "Homo_sapiens":
                write_clw_true_state_seq(stripped_fasta_file_path, out_dir_path = exon_dir)

        # gather all usable fasta seqs in a single file
        input_files = get_input_files_with_human_at_0(from_path = stripped_seqs_dir)

        output_file = os.path.join(exon_dir, "combined.fasta")
        combine_fasta_files(output_file = output_file, input_files = input_files)

        if args.convert_short_lc_to_UC > 0:
            output_file = os.path.join(capitalzed_subs_seqs_dir, "combined.fasta")
            convert_short_lc_to_UC(output_file, input_files, threshold = args.convert_short_lc_to_UC)



########################################################################################################################
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
            exon_coords = list(map(int, exon.split("_")[2:]))
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
    args = parser.parse_args()

    assert args.left_exon_anchor_len < args.min_left_exon_len, \
        "[ERROR] >>> left_exon_anchor_len > min_left_exon_len"
    assert args.right_exon_anchor_len < args.min_right_exon_len, \
        "[ERROR] >>> right_exon_anchor_len > min_right_exon_len"
    assert args.middle_exon_anchor_len < args.min_exon_len, \
        "[ERROR] >>> middle_exon_anchor_len > min_exon_len"
    # TODO im using the above also for the start and end of th middle exon, not only the middle of the middle/current exon

    if not args.stats_table:
        assert args.hg38 and args.hal and args.species, "you must pass path to hg38, hal and species.lst"
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

        hg38_refseq_bed = load_hg38_refseq_bed()
        filtered_internal_exons = get_to_be_lifted_exons(hg38_refseq_bed, json_path, overwrite)
        create_exon_data_sets(filtered_internal_exons, output_dir)
        make_stats_table()
    else:
        if not os.path.isdir(args.stats_table):
            assert args.hg38 and args.hal and args.species, \
                "you must pass path to hg38, hal and species.lst or path to the dir of which the stats table should \
                    get created"
        make_stats_table()
