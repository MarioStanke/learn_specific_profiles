""" Data classes and named tuples for better IDE integration """

from dataclasses import dataclass
from collections import namedtuple

# define named tuple for exon to make it safer to use in a proper editor/IDE
# Exon = namedtuple("Exon", ["seq", "start_in_genome", "stop_in_genome", "id", 
#                            "left_anchor_start", "left_anchor_end", "right_anchor_start", "right_anchor_end",
#                            "left_intron_len", "right_intron_len", "strand", "substring_len", "total_len"])

# use a dataclass instead
@dataclass
class Exon:
    seq: str
    start_in_genome: int
    stop_in_genome: int
    id: int
    left_anchor_start: int
    left_anchor_end: int
    right_anchor_start: int
    right_anchor_end: int
    left_intron_len: int
    right_intron_len: int
    strand: str
    substring_len: int
    total_len: int

    # return object as dict
    def as_dict(self):
        return self.__dict__

# for the same reason, create simple class for liftover_seq
@dataclass
class LiftoverSeq:
    seq_start_in_genome: int = None
    seq_stop_in_genome: int = None
    middle_of_exon_start: int = None
    middle_of_exon_stop: int = None
    on_reverse_strand: bool = None
    seq_name: str = None
    substring_len: int = None # without anchors

# same reason, must match the named tuple returned by Pandas itertuples() method, check this later
BedRow = namedtuple("BedRow", ["Index", "chrom", "chromStart", "chromEnd", "name", "score", "strand", 
                               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"])
BedColumns = BedRow._fields[1:] # chrom, chromStart, chromEnd, ...