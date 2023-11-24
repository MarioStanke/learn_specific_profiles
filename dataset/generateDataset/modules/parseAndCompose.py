""" Functions for parsing and composing data like GTF etc. """

import pandas as pd

from .dtypes import BedColumns, BedRow, Exon

def parse_bed(bedfile) -> pd.DataFrame:
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
    if ncol == 12:
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

    # check tuple names
    if ncol == 12:
        for row in bedf.itertuples():
            # assert that row matches the BedRow namedtuple, this allows to hint the type of the row as BedRow
            assert row._fields == BedRow._fields, "[ERROR] >>> BedRow namedtuple fields do not match hg38_refseq_bed " \
                + f"columns: {BedRow._fields} != {row._fields}"
            break # one iteration is enough

    return bedf



def parseGTF(gtf_file, seqname = None, range_start = None, range_end = None, 
             source = None, feature = None, strand = None, min_score = None, max_score = None, 
             frame = None) -> pd.DataFrame:
    """ Load a GTF file into a Pandas DataFrame and return it. The returned DataFrame does _not_ follow the GTF position
        logic, i.e. positions are 0-based and range_end is exclusive. The returned DataFrame has the following columns:
            seqname: the seqname
            source: the source
            feature: the feature
            start: the start position
            end: the end position
            score: the score
            strand: the strand
            frame: the frame
            attribute: the attribute

        Note: range_start and range_end arguments also use the BED-file logic, i.e. positions are 0-based and range_end 
              is exclusive.
        Also note: This parser cannot handle track lines or other non-data lines in the GTF file.

        The optional arguments can be used to filter the 
        DataFrame:
            seqname: only return entries that match the seqname
            range_start: only return entries that _end_ after range_start (partial overlap is allowed)
            range_end: only return entries that _start_ before range_end (partial overlap is allowed)
            source: only return entries that match the source
            feature: only return entries that match the feature
            strand: only return entries that match the strand
            min_score: only return entries that have a score >= min_score
            max_score: only return entries that have a score <= max_score
            frame: only return entries that match the frame
    """	
    assert range_start is None or range_start >= 0, "[ERROR] >>> range_start < 0"
    assert range_end is None or range_end >= 0, "[ERROR] >>> range_end < 0"
    assert range_start is None or range_end is None or range_start < range_end, "[ERROR] >>> range_start >= range_end"
    assert strand is None or strand in ["+", "-"], "[ERROR] >>> strand must be None, '+' or '-'"
    assert min_score is None or max_score is None or min_score <= max_score, "[ERROR] >>> min_score > max_score"
    assert frame is None or frame in [0, 1, 2], "[ERROR] >>> frame must be None, 0, 1 or 2"
    
    # Info: in GTF, positions are 1-based and ranges include both start and end, i.e. a range [1,2] includes the first
    #       and second base of the sequence.
    gtf_range_start = range_start + 1 if range_start is not None else None
    gtf_range_end = range_end if range_end is not None else None

    gtf = pd.read_csv(gtf_file, sep = "\t", header = None, index_col = None,
                      names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
                      na_values=".", comment="#")
    
    # filter by seqname
    if seqname is not None:
        gtf = gtf[gtf["seqname"] == seqname]
    # filter by source
    if source is not None:
        gtf = gtf[gtf["source"] == source]
    # filter by feature
    if feature is not None:
        gtf = gtf[gtf["feature"] == feature]
    # filter by strand
    if strand is not None:
        gtf = gtf[gtf["strand"] == strand]
    # filter by score
    if min_score is not None:
        gtf = gtf[gtf["score"] >= min_score]
    if max_score is not None:
        gtf = gtf[gtf["score"] <= max_score]
    # filter by frame
    if frame is not None:
        gtf = gtf[gtf["frame"] == frame]
    # filter by range
    if range_start is not None:
        gtf = gtf[gtf["end"] >= gtf_range_start]
    if range_end is not None:
        gtf = gtf[gtf["start"] <= gtf_range_end]

    # convert start and end to 0-based positions and end-exclusive ranges
    gtf["start"] = gtf["start"] - 1

    return gtf



def create_human_liftover_bed(exon: Exon, out_path, args):
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