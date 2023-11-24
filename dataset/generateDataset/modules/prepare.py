""" Prepare everything before generating the data via liftover. """

import json
import logging
import numpy as np
import os
import pandas as pd
import time

from .dtypes import Exon, BedRow

########################################################################################################################
# def load_hg38_refseq_bed():
#     """ Load hg38 refseq bed file (specified in arguments) as Pandas DataFrame and return it. """
#     start = time.perf_counter()
#     print("[INFO] >>> started load_hg38_refseq_bed()")
#     hg38_refseq_bed = pd.read_csv(args.hg38, delimiter = "\t", header = None)
#     assert hg38_refseq_bed.columns.size == 12, \
#         f"[ERROR] >>> hg38 refseq bed has {hg38_refseq_bed.columns.size} columns, 12 expected"
#     hg38_refseq_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", 
#                                "itemRgb", "blockCount", "blockSizes", "blockStarts"]
#     def parseBlockList(s):
#         fields = s.split(",")
#         if fields[-1] == '':
#             return [int(a) for a in fields[:-1]]
#         else:
#             return [int(a) for a in fields]

#     hg38_refseq_bed["blockSizes"] = hg38_refseq_bed["blockSizes"].map(parseBlockList)
#     hg38_refseq_bed["blockStarts"] = hg38_refseq_bed["blockStarts"].map(parseBlockList)
#     assert all(hg38_refseq_bed["blockCount"] == hg38_refseq_bed["blockSizes"].map(len)), \
#         "[ERROR] >>> blockCount != len(blockSizes)"
#     assert all(hg38_refseq_bed["blockCount"] == hg38_refseq_bed["blockStarts"].map(len)), \
#         "[ERROR] >>> blockCount != len(blockStarts)"

#     # just to be safe
#     def ensureNumericColumn(column):
#         if not pd.api.types.is_numeric_dtype(hg38_refseq_bed[column]):
#             hg38_refseq_bed[column] = pd.to_numeric(hg38_refseq_bed[column])

#         return hg38_refseq_bed
    
#     hg38_refseq_bed = ensureNumericColumn("chromStart")
#     hg38_refseq_bed = ensureNumericColumn("chromEnd")
#     hg38_refseq_bed = ensureNumericColumn("score")
#     hg38_refseq_bed = ensureNumericColumn("thickStart")
#     hg38_refseq_bed = ensureNumericColumn("thickEnd")
#     hg38_refseq_bed = ensureNumericColumn("blockCount")

#     print("[INFO] >>> finished load_hg38_refseq_bed(). It took:", time.perf_counter() - start)
#     return hg38_refseq_bed
########################################################################################################################



def get_all_internal_exons(hg38_refseq_bed: pd.DataFrame, args):
    """ Get all internal exons from the given hg38 refseq bed file and return them as a dictionary where the keys are
        (chromosome, exon start, exon stop) and the values are lists of rows that mapped to this exon range. """
    start = time.perf_counter()
    logging.debug("Started get_all_internal_exons()")
    internal_exons = {} # keys: (chromosome, exon start, exon stop), values: list of rows that mapped to this exon range

    dbg_internal_exons_NM = {} # [DEBUG] >>> same as internal_exons but only for NM_ genes

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
        # [DEBUG] >>> also only allow exons from regular chromosomes, not chrA_alt_* etc.
        allowed_chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
        if chromosome not in allowed_chromosomes:
            continue

        for exon_len, exon_start in zip(row.blockSizes[1:-1], row.blockStarts[1:-1]):
            exon_start_in_genome = row.chromStart + exon_start
            exon_end_in_genome = row.chromStart + exon_start + exon_len # the end pos is not included
            key = (chromosome, exon_start_in_genome, exon_end_in_genome)
            if key not in internal_exons:
                internal_exons[key] = []

            internal_exons[key].append(row)

            # [DEBUG] >>> only add exons from NM_ genes
            if row.name.startswith("NM_"):
                if key not in dbg_internal_exons_NM:
                    dbg_internal_exons_NM[key] = []

                dbg_internal_exons_NM[key].append(row)

    # [DEBUG] >>> compare internal_exons and dbg_internal_exons_NM
    logging.debug(f"len(internal_exons) = {len(internal_exons)}, len(dbg_internal_exons_NM) = {len(dbg_internal_exons_NM)}")
    logging.debug(f"average number of rows per exon in internal_exons = {sum([len(v) for v in internal_exons.values()])/len(internal_exons)}")
    logging.debug(f"average number of rows per exon in dbg_internal_exons_NM = {sum([len(v) for v in dbg_internal_exons_NM.values()])/len(dbg_internal_exons_NM)}")

    # [DEBUG] >>> print rows that mapped to the same exon
    #i = 0
    #for key in dbg_internal_exons_NM.keys():
    #    if len(dbg_internal_exons_NM[key]) > 1:
    #        print("[DEBUG] >>> key is exon = ", key)
    #        print("  value is list of df_rows:")
    #        for df_row in dbg_internal_exons_NM[key]:
    #            print("   ", df_row)
    #
    #        print()
    #
    #    i += 1
    #    if i >= 2:
    #        break

    if args.v:
        logging.info("Since the same exon occurs in multiple genes, which may just be spliced differently") # wat?
        for key in sorted(internal_exons.keys()):
            if len(internal_exons[key]) > 3:
                logging.info(f"  key is exon = {key}")
                logging.info("  value is list of df_rows:")
                for df_row in internal_exons[key]:
                    logging.info(f"    {df_row}")
                break

    logging.debug(f"Finished get_all_internal_exons(). It took {time.perf_counter() - start}")
    #return internal_exons
    return dbg_internal_exons_NM # [DEBUG] >>> only return NM_ genes



def filter_and_choose_exon_neighbours(all_internal_exons: dict[tuple, list[BedRow]], args) -> list[Exon]:
    """ Because of alternatively spliced genes, an exon might have multiple neighboring exons to choose from. 
        Currently, simply the first splice form that matches the requirements is chosen for each exon. """
    start = time.perf_counter()
    logging.debug("Started filter_and_choose_exon_neighbours()")
    filtered_internal_exons = []

    # shuffle the keys to avoid only using Chr1 exons
    keys = list(all_internal_exons.keys())
    logging.debug(f"Unshuffled keys: {keys[:10]}")
    rng = np.random.default_rng(42) # use seed to always get the same behaviour
    rng.shuffle(keys)
    logging.debug(f"Shuffled keys: {keys[:10]}")

    #for key in all_internal_exons.keys():
    for key in keys:
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
                logging.warning(f"exon_id == {exon_id} for exon {key} in row {row}")
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
            logging.info(f"filtered_internal_exons[{i}]")
            for key in sorted(filtered_internal_exons[i].keys()):
                logging.info(f"  {key} {filtered_internal_exons[i][key]}")
    logging.debug(f"Finished filter_and_choose_exon_neighbours(). It took {time.perf_counter() - start}")
    return filtered_internal_exons



def write_filtered_internal_exons(filtered_internal_exons: list[Exon], json_path):
    """ Store the filtered internal exons in a json file for later use. Avoid recomputing them. """
    start = time.perf_counter()
    logging.debug("Started write_filtered_internal_exons()")
    # convert Exon objects to dicts for json.dump()
    exon_list = [e.as_dict() for e in filtered_internal_exons]
    with open(json_path, "w") as json_out:
        json.dump(exon_list, json_out, indent=2)
    logging.debug(f"Finished write_filtered_internal_exons(). It took {time.perf_counter() - start}")



def get_to_be_lifted_exons(hg38_refseq_bed, json_path, overwrite, args) -> list[Exon]:
    """ Either load the filtered internal exons from a json file or compute them: Only consider exons that are internal,
         i.e. have at least one exon on the left and one on the right. Then, apply length filters to the exons and their
         neighbouring introns and exons. If desired, apply a cutoff of `args.n` to thee final list of exons. """
    if os.path.exists(json_path) and not overwrite:
        logging.debug(f"The file {json_path} exists, so it isn't computed again")
        start = time.perf_counter()
        logging.debug(f"Started json.load({json_path})")
        with open(json_path) as file:
            filtered_internal_exons = json.load(file)
        logging.debug(f"Finished json.load({json_path}). It took {time.perf_counter() - start}")
    else:
        all_internal_exons = get_all_internal_exons(hg38_refseq_bed, args)
        filtered_internal_exons = filter_and_choose_exon_neighbours(all_internal_exons, args)
        write_filtered_internal_exons(filtered_internal_exons, json_path)
    if args.n and len(filtered_internal_exons) > args.n:
        filtered_internal_exons = filtered_internal_exons[:args.n]

    return filtered_internal_exons
