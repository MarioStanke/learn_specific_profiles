""" Perform the steps to create the dataset. """

from Bio import Align, AlignIO, Seq, SeqIO, SeqRecord
from collections import namedtuple
import json
import logging
import math
import multiprocessing
import os
import pandas as pd
import re
import shutil
import subprocess
import time

from .dtypes import Exon, LiftoverSeq
from . import mplog
from .parseAndCompose import parse_bed, parseGTF, create_human_liftover_bed

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "modules"))
import SequenceRepresentation



def liftover(human_exon_to_be_lifted_path, species_name, out_dir, exon: Exon, args, 
             logger: mplog.customBufferedLogger): # DEBUG: exon argument
    """ Either creates new bed file by calling halLiftover with the human exon bed file and the species name,
        or searches for an old  (only if --use_old_bed is given). 
        Returns named tuple with returncode that can be checked for success (returncode 0) """
    bed_file_path = os.path.join(out_dir, species_name+".bed")
    if not args.use_old_bed:
        command = f"time halLiftover {args.hal} Homo_sapiens {human_exon_to_be_lifted_path} {species_name} " \
                    + f"{bed_file_path}"
        logger.debug(f"running: {command}")
        status = subprocess.run(command, shell=True, capture_output=True)
        #os.system(command)

        # ~~~ DEBUG ~~~
        # checking that halLiftover does what is expected by comparing the generated BED for human with the input
        # --> it should be the same, otherwise die
        if species_name == "Homo_sapiens":
            die = False
            logger.debug("Homo sapiens self liftover")
            logger.debug(f"  Exon: {exon}")
            # create fasta from exon, human_exon_bed and lifted bed
            exonfasta = os.path.join(out_dir, "DEBUG_human_exon.fasta")
            bedfasta = os.path.join(out_dir, "DEBUG_human_bed.fasta")
            liftfasta = os.path.join(out_dir, "DEBUG_human_lift.fasta")
            run_hal_2_fasta(species_name, exon.left_anchor_start, exon.total_len, exon.seq, exonfasta, args, logger)
            
            hgbed = parse_bed(human_exon_to_be_lifted_path)
            assert len(hgbed.index) == 3, f"[ERROR] >>> human exon bedfile has {len(hgbed.index)} lines (3 expected)"
            assert hgbed.iloc[0]['name'][-4:] == 'left', "[ERROR] >>> human exon bedfile not as expected:\n"+str(hgbed)
            assert hgbed.iloc[1]['name'][-5:] == 'right', "[ERROR] >>> human exon bedfile not as expected:\n"+str(hgbed)
            bedstart = hgbed.iloc[0]['chromStart']
            bedend = hgbed.iloc[1]['chromEnd']
            assert bedstart < bedend, "[ERROR] >>> left start after right end:\n"+str(hgbed)
            bedlen = bedend-bedstart
            run_hal_2_fasta(species_name, bedstart, bedlen, exon.seq, bedfasta, args, logger)

            try:
                liftbed = parse_bed(bed_file_path)
            except:
                logger.error(f"[liftover() debug section] Could not parse bed file {bed_file_path}" + \
                              " (probably empty), skipping further debugging")
                return status # skip the rest but also don't die
                
            assert len(liftbed.index) == 3, f"[ERROR] >>> lifted bedfile has {len(liftbed.index)} lines (3 expected)"
            assert liftbed.iloc[0]['name'][-4:] == 'left', "[ERROR] >>> lifted bedfile not as expected:\n"+str(liftbed)
            assert liftbed.iloc[1]['name'][-5:] == 'right', "[ERROR] >>> lifted bedfile not as expected:\n"+str(liftbed)
            bedstart = liftbed.iloc[0]['chromStart']
            bedend = liftbed.iloc[1]['chromEnd']
            assert bedstart < bedend, "[ERROR] >>> left start after right end:\n"+str(liftbed)
            bedlen = bedend-bedstart
            run_hal_2_fasta(species_name, bedstart, bedlen, exon.seq, liftfasta, args, logger)

            # load generated fastas and compare (should all be the same, 
            #   especially lifted over sequence from human to human)
            exonseq = []
            with open(exonfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    exonseq.append(record)

            logger.debug(f"  number of exon sequences: {len(exonseq)}")
            logger.debug("  Exon sequence:")
            logger.debug(f"  {exonseq[0]}")

            bedseq = []
            with open(bedfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    bedseq.append(record)

            logger.debug(f"  number of bed sequences: {len(bedseq)}")
            logger.debug("  BED sequence:")
            logger.debug(f"  {bedseq[0]}")

            liftseq = []
            with open(liftfasta) as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    liftseq.append(record)

            logger.debug(f"  number of lifted sequences: {len(liftseq)}")
            logger.debug("  Lifted sequence:")
            logger.debug(f"  {liftseq[0]}")

            def logStackedSeqs(seq1: str, seq2: str, lw=20):
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
                    logger.warning(s1)
                    logger.warning(m)
                    logger.warning(s2)
                    logger.warning("")

                    a1 = b1
                    a2 = b2



            if exonseq[0].seq != bedseq[0].seq:
                logger.warning("Exon seq and BED seq differ")
                logStackedSeqs(exonseq[0].seq, bedseq[0].seq, 120)
                die = True
            else:
                logger.debug("Exon seq and BED seq are the same")

            if exonseq[0].seq != liftseq[0].seq:
                logger.warning("Exon seq and lifted seq differ")
                logStackedSeqs(exonseq[0].seq, liftseq[0].seq, 120)
                die = True
            else:
                logger.debug("Exon seq and lifted seq are the same")

            if bedseq[0].seq != liftseq[0].seq:
                logger.warning("BED seq and lifted seq differ")
                logStackedSeqs(bedseq[0].seq, liftseq[0].seq, 120)
                die = True
            else:
                logger.debug("BED seq and lifted seq are the same")

            #print("[DEBUG] >>> Lifted BED:")
            #print(liftbed)
            if die:
                logger.critical("Exiting due to errors")
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
    


def extract_info_and_check_bed_file(bed_dir, species_name, exon: Exon, liftover_seq: LiftoverSeq, args, 
                                    logger: mplog.customBufferedLogger):
    """ Checks the generated bedfile from halLiftover and extracts sequence information into liftover_seq.
        Sequence coordinates exclude the left and right anchors.
        Returns true if everything was successful, false if something was wrong with the bed file. """
    bed_file_path = os.path.join(bed_dir, species_name+".bed")
    if os.path.getsize(bed_file_path) == 0:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} is empty")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_empty.bed"))
        return False

    species_bed = parse_bed(bed_file_path)
    assert species_bed.columns.size == 6, \
        f"[ERROR] >>> Bed file {bed_file_path} has {species_bed.columns.size} columns, 6 expected"
    #species_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]

    if len(species_bed.index) != 3 and args.discard_multiple_bed_hits:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} has {len(species_bed.index)} lines, 3 expected " + \
                     f"because args.discard_multiple_bed_hits is set to {args.discard_multiple_bed_hits}")
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
            logger.critical(f"An error occured: {e}")
            logger.critical("l_m_r[x.group(1)] didnt work")
            logger.critical(f"            row.name {row.name}")
            logger.critical(f"            bed_file_path {bed_file_path}")
            exit(1)
        # Exception might miss some errors, so catch all errors
        except:
            logger.critical("l_m_r[x.group(1)] didnt work")
            logger.critical(f"            row.name {row.name}")
            logger.critical(f"            bed_file_path {bed_file_path}")
            exit(1)

    if len(l_m_r) != 3:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} does not contain longest hits for each " + \
                     f"left, middle and right: {len(l_m_r)} longest hits, 3 expected")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_not_all_l_m_r.bed"))
        return False
    if l_m_r["left"]["strand"] != l_m_r["right"]["strand"] or l_m_r["left"]["strand"] != l_m_r["middle"]["strand"]:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} longest hits for left, middle and right not all on "+\
                     "same strand")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_unequal_strands.bed"))
        return False
    if l_m_r["left"]["chrom"] != l_m_r["left"]["chrom"] or l_m_r["left"]["chrom"] != l_m_r["middle"]["chrom"]:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} longest hits for left, middle and right not all on "+\
                     "same chromosome")
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
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} seq_start_in_genome >= middle_of_exon_start")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_left_greater_middle.bed"))
        return False
    if liftover_seq.middle_of_exon_stop >= liftover_seq.seq_stop_in_genome:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} middle_of_exon_stop >= seq_stop_in_genome")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_right_less_middle.bed"))
        return False

    liftover_seq.on_reverse_strand = (l_m_r["left"]["strand"] == "-")
    liftover_seq.seq_name = l_m_r['left']['chrom']
    liftover_seq.substring_len = liftover_seq.seq_stop_in_genome - liftover_seq.seq_start_in_genome

    threshold = 1
    l1 = liftover_seq.substring_len
    l2 = exon.substring_len
    if abs(math.log10(l1) - math.log10(l2)) >= threshold:
        logger.debug(f"[LIFTOVER ERROR] Bed file {bed_file_path} lengths differ by more than {threshold} orders " + \
                     f"of magnitude: {l1} vs {l2}")
        shutil.move(bed_file_path, os.path.join(bed_dir, species_name+"_errorcode_lengths_differ_substantially.bed"))
        return False

    return True



def write_extra_data_to_fasta_description_and_reverse_complement(fa_path,liftover_seq: LiftoverSeq, exon: Exon,
                                                                 logger: mplog.customBufferedLogger):
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
        logger.warning(f"No seq found in fasta file {fa_path}, skipping")
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

    # # Create a SequenceRepresentation.Sequence object from he Exon and LiftoverSeq objects and store it, this should
    # # contain all relevant information for the evaluation of the profile finding
    # hgSeq = SequenceRepresentation.Sequence(species="Homo_sapiens", chromosome=exon.seq, strand=exon.strand,
    #                                         genome_start=exon.left_anchor_start, genome_end=exon.right_anchor_end,
    #                                         no_homology=True)
    # hgSeq.addSubsequenceAsElement(exon.left_anchor_start, exon.left_anchor_end, 'left_anchor', genomic_positions=True,
    #                               no_elements=True, no_homology=True)
    # hgSeq.addSubsequenceAsElement(exon.right_anchor_start, exon.right_anchor_end, 'right_anchor', 
    #                               genomic_positions=True, no_elements=True, no_homology=True)
    # hgSeq.addSubsequenceAsElement(exon.start_in_genome, exon.stop_in_genome, 'exon', genomic_positions=True,
    #                               no_elements=True, no_homology=True)
    
    # liftstrand = "-" if liftover_seq.on_reverse_strand else "+"
    # liftSeq = SequenceRepresentation.Sequence(species=species, chromosome=liftover_seq.seq_name, strand=liftstrand,
    #                                           genome_start=liftover_seq.seq_start_in_genome, 
    #                                           genome_end=liftover_seq.seq_stop_in_genome)
    # liftSeq.addHomology(hgSeq)
    # # TODO: scan annotation file of species for the exon and add candidates to the liftSeq object

    # with open(seqpath, 'wt') as fh:
    #     json.dump(liftSeq.toDict(), fh, indent=4)



def gtf_to_sequence_elements(sequence: SequenceRepresentation.Sequence, gtf: pd.DataFrame):
    """ Turns each line in a GTF DataFrame into an element of the sequence. Expects `gtf` to be already filtered and
        only contain valid lines that are actually conatined in `sequence` """
    
    for row in gtf.itertuples():
        assert sequence.chromosome == row.seqname, f"[ERROR] >>> sequence.chromosome ({sequence.chromosome}) != " \
                                                                                        + f"row.seqname ({row.seqname})"
        
        # handle the source attribute: if possible, use a dict, otherwise the string representation of the row
        try:
            source = row._asdict()
            json.dumps(source) # check if source can be converted to json
        except:
            source = str(row)

        # create a new element
        sequence.addSubsequenceAsElement(row.start, row.end, row.feature, row.strand, source=source,
                                         genomic_positions=True, no_elements=True)



def create_sequence_representation_object(fa_path: str, species: str, liftover_seq: LiftoverSeq, 
                                          exon: Exon, 
                                          logger: mplog.customBufferedLogger) -> SequenceRepresentation.Sequence:
    """ Reads single sequence fasta from fa_path and creates a SequenceRepresentation.Sequence object from the Exon and
        LiftoverSeq objects and returns it, this should contain all relevant information for the evaluation of 
        profile_finding """
    # read fasta file
    sequence = None
    for seq in SeqIO.parse(fa_path, "fasta"):
        if sequence is not None:
            logger.warning(f"Found more than one seq in fasta file {fa_path}, skipping")
            return
        
        # has been converted to reverse complement before if liftover_seq is on reverse strand, undo that here
        sequence = str(seq.seq) if not liftover_seq.on_reverse_strand else (str(seq.seq.reverse_complement()))

    if sequence is None:
        logger.warning(f"No seq found in fasta file {fa_path}, skipping")
        return

    # Create a SequenceRepresentation.Sequence object from the Exon and LiftoverSeq objects and store it, this should
    # contain all relevant information for the evaluation of the profile finding
    hgSeq = SequenceRepresentation.Sequence(species="Homo_sapiens", chromosome=exon.seq, strand=exon.strand,
                                            genome_start=exon.left_anchor_start, genome_end=exon.right_anchor_end,
                                            no_homology=True)
    hgSeq.addSubsequenceAsElement(exon.left_anchor_start, exon.left_anchor_end, 'left_anchor', genomic_positions=True,
                                  no_elements=True, no_homology=True)
    hgSeq.addSubsequenceAsElement(exon.right_anchor_start, exon.right_anchor_end, 'right_anchor', 
                                  genomic_positions=True, no_elements=True, no_homology=True)
    hgSeq.addSubsequenceAsElement(exon.start_in_genome, exon.stop_in_genome, 'exon', genomic_positions=True,
                                  no_elements=True, no_homology=True)
    
    liftstrand = "-" if liftover_seq.on_reverse_strand else "+"
    liftSeq = SequenceRepresentation.Sequence(species=species, chromosome=liftover_seq.seq_name, strand=liftstrand,
                                              genome_start=liftover_seq.seq_start_in_genome, 
                                              genome_end=liftover_seq.seq_stop_in_genome, sequence=sequence)
    liftSeq.addHomology(hgSeq)
    
    logger.debug(f"Adding annotation to sequence representation object for species {species}")
    annotfile = os.path.join("/home/ebelm/genomegraph/data/241_species/annot" , f"{species}.gtf")
    if os.path.exists(annotfile):
        # ADDITIONAL FILTERING OF HUMAN ANNOTATION
        source = "BestRefSeq" if species == 'Homo_sapiens' else None # only use high-quality annotation for human
        annot = parseGTF(annotfile, seqname=liftSeq.chromosome, range_start=liftSeq.genome_start, 
                         range_end=liftSeq.genome_end, feature="CDS", source=source)
        gtf_to_sequence_elements(liftSeq, annot)
    logger.debug(f"Done adding annotation to sequence representation object for species {species}")
    #print("[DEBUG] >>> sequence representation object for species", species, ":", liftSeq.toDict())

    return liftSeq



def run_hal_2_fasta(species_name, start, len, seq, outpath, args, logger: mplog.customBufferedLogger):
    """ Executes shell command hal2fasta that extracts a subsequence from the genome of a species in the hal-file and 
        prints the first 10 lines of the output fasta file """
    command = f"time hal2fasta {args.hal} {species_name} --start {start} --length {len} --sequence {seq} \
                --ucscSequenceNames --outFaPath {outpath}"
    logger.debug(f"Running {command}")
    status = subprocess.run(command, shell=True, capture_output=True)
    if not status.returncode == 0:
        logger.warning(f"hal2fasta failed with error code {status.returncode}")
        logger.warning(status.stderr.decode("utf-8"))
    else:
        #os.system(command)
        logger.debug(f"Running: head {outpath}")
        status = subprocess.run(f"head -n 5 {outpath}", shell=True, capture_output=True)
        logger.debug(status.stdout.decode("utf-8"))
        #os.system(f"head {outpath}")



# TODO: can I maybe use the old fasta description such that i dont have to pass liftover_seq
def strip_seqs(fasta_file, exon: Exon, out_path, liftover_seq: LiftoverSeq, args):
    """ Uses the liftover fasta files. There, the left and right anchors are already removed.
        If fasta_file exists, strips the sequence in the fasta file and writes the stripped sequence to `out_path`.
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



def capitalize_lowercase_subseqs(seq, threshold):
        # (?<![a-z]) negative lookbehind, (?![a-z]) negative lookahead, i.e. only match if the character before and 
        #   after the match is not a lowercase letter
        pattern = f"(?<![a-z])([a-z]{{1,{threshold}}})(?![a-z])" 
        def repl(match):
            return match.group(1).upper()
        result = re.sub(pattern, repl, seq)
        return str(result)



def convert_short_lc_to_UC(outpath, input_files, threshold):
    """ Converts short (up to threshold length) softmasked parts of the sequences in input_files to upper case and
        append the results to outpath. The sequences in input_files are assumed to be in fasta format. """    
    # TODO this looks horrible, fix it and use SeqIO
    with open(outpath, "wt") as output_handle:
        for input_file in input_files:
            with open(outpath, "a") as output_handle, open(input_file, "r") as in_file_handle:
                for i, record in enumerate(SeqIO.parse(in_file_handle, "fasta")):
                    assert i == 0, f"convert_short_lc_to_UC found more than one seq in fasta file {input_file}"
                    new_seq = capitalize_lowercase_subseqs(str(record.seq), threshold)
                    output_handle.write(f">{record.id} {record.description}\n")
                    output_handle.write(f"{new_seq}\n")
                    # new_record = record.__class__(seq = "ACGT", id="homo", name="name", description="description")
                    # SeqIO.write(new_record, output_handle, "fasta")
                    # this produced
                    # File "/usr/lib/python3/dist-packages/Bio/File.py", line 72, in as_handle
                    # with open(handleish, mode, **kwargs) as fp:



def get_input_files_with_human_at_0(from_path, logger: mplog.customBufferedLogger):
    """ Gets all file paths of .fa file from from_path, returns them as a list with the Homo_sapiens fasta as first
        element. """
    input_files = [os.path.join(from_path, f) for f in os.listdir(from_path) if f.endswith(".fa")]
    input_files = sorted(input_files, key = lambda x: 0 if re.search("Homo_sapiens", x) else 1)
    #assert re.search("Homo_sapiens", input_files[0]), f"[ERROR] >>> Homo sapiens not in first pos of {from_path}"
    if not re.search("Homo_sapiens", input_files[0]):
        logger.error(f"Homo sapiens not in first pos of {from_path}")

    return input_files



def combine_fasta_files(output_file, input_files):
    """ Write all sequences from the fasta files in input_files to output_file """
    # TODO this looks weird, also out.close() should be unneccessary, refactor this
    with open(output_file, "wt") as out:
        for input_file in input_files:
            for seq_record in SeqIO.parse(input_file, "fasta"):
                SeqIO.write(seq_record, out, "fasta")
    out.close()



def write_clw_true_state_seq(fasta_path, out_dir_path, logger: mplog.customBufferedLogger):
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
            logger.error(f"No human id found in {fasta_path}")
            return
    except:
        logger.error("SeqIO could not parse {fasta_path}")
        return

    try:
        coords = json.loads(re.search("({.*})", human_fasta.description).group(1))
    except:
        logger.critical(f"Could not extract coordinates from fasta header {human_fasta.description}")
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
    logger.debug(f"Wrote alignment to {alignment_out_path}")



def create_exon_data_sets(filtered_internal_exons: list[Exon], output_dir, args):
    """ For each exon in filtered_internal_exons, computes liftover for each species and stores all sequences in a
        fasta file """
    with open(args.species, "rt") as species_file:
        species = species_file.readlines()

    all_species = [s.strip() for s in species]

    # setup logging
    logqueue = multiprocessing.Queue(-1)
    loglistener = multiprocessing.Process(target=mplog.mp_log_listener, args=(logqueue,))
    loglistener.start() # process continously checking the queue for new messages

    processes = []
    for exon_i, exon in enumerate(filtered_internal_exons):
        p = multiprocessing.Process(target=run_exon_creation, args = (exon, all_species, output_dir, exon_i,
                                                                      len(filtered_internal_exons), args, logqueue))
        processes.append(p)

    # start processes and wait for them to finish
    mplog.mp_process_manager(processes, args.num_processes)
    
    logqueue.put(None) # tell the listener to quit
    loglistener.join() # wait for the listener to quit

    #for exon_i, exon in enumerate(filtered_internal_exons):



# put into own function for multiprocessing
def run_exon_creation(exon, all_species, output_dir, exon_i, total_i, args, logqueue):
    """ Run exon creation for a single exon: computes liftover for each species and stores all sequences in a
        fasta file """
    start = time.perf_counter()

    logger = mplog.customBufferedLogger()
    logger.info(f"Processing exon {exon_i+1}/{total_i}: " + \
                    f"{exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}")
    exon_dir = os.path.join(output_dir, f"exon_{exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}")
    bed_output_dir = os.path.join(exon_dir, "species_bed")
    seqs_dir = os.path.join(exon_dir, "species_seqs")
    non_stripped_seqs_dir = os.path.join(seqs_dir, "non_stripped")
    stripped_seqs_dir = os.path.join(seqs_dir, "stripped")
    capitalzed_subs_seqs_dir = os.path.join(exon_dir, f"combined_fast_capitalized_{args.convert_short_lc_to_UC}")
    
    for d in [exon_dir, bed_output_dir, seqs_dir, non_stripped_seqs_dir, 
                stripped_seqs_dir, capitalzed_subs_seqs_dir]:
        os.makedirs(d, exist_ok = True)

    # Quick fix to avoid reruns after failed jobs TODO do this properly
    output_file = os.path.join(capitalzed_subs_seqs_dir, "combined.fasta") \
        if args.convert_short_lc_to_UC > 0 else os.path.join(exon_dir, "combined.fasta")
    if os.path.isfile(output_file):
        logger.warning(f"Skipping exon {exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}" + \
                        f"because {output_file} already exists")
        return

    human_exon_bedfile = os.path.join(exon_dir, "human_exons.bed")
    create_human_liftover_bed(exon = exon, out_path = human_exon_bedfile, args = args)
    sequence_reps = [] # store all sequence representations here
    error_species = 0
    for single_species in all_species:
        bed_status = liftover(human_exon_bedfile, single_species, bed_output_dir, exon, args, logger)
        if bed_status.returncode == 1:
            logger.warning(f"No bed file for species {single_species}: {bed_status}")
            continue

        liftover_seq = LiftoverSeq()
        if not extract_info_and_check_bed_file(
            bed_dir = bed_output_dir,
            species_name = single_species,
            exon = exon,
            liftover_seq = liftover_seq,
            args = args,
            logger = logger
        ):
            error_species += 1
            continue


        # getting the seq, from human: [left exon [lifted]] [intron] [exon] [intron] [[lifted] right exon]
        # the corresponding seq of [intron] [exon] [intron] in other species
        out_fa_path = os.path.join(non_stripped_seqs_dir, single_species+".fa")
        if not args.use_old_fasta:
            run_hal_2_fasta(species_name = single_species,
                            start = liftover_seq.seq_start_in_genome,
                            len = liftover_seq.substring_len,
                            seq = liftover_seq.seq_name,
                            outpath = out_fa_path,
                            args = args,
                            logger = logger)

            write_extra_data_to_fasta_description_and_reverse_complement(fa_path = out_fa_path,
                                                                         liftover_seq = liftover_seq,
                                                                         exon = exon,
                                                                         logger = logger)
            
        # [ME] here I gather the information for my pipeline, the remaining steps do not affect it
        sr = create_sequence_representation_object(out_fa_path, single_species, liftover_seq, exon, logger)
        if sr is not None:
            sequence_reps.append(sr)
        # ----------------------------------------------------------------------------------------

        # strip short segments from the beginning and end of the sequences (why though?)
        stripped_fasta_file_path = re.sub("non_stripped","stripped", out_fa_path)
        strip_seqs(fasta_file = out_fa_path,
                    exon = exon,
                    out_path = stripped_fasta_file_path,
                    liftover_seq = liftover_seq,
                    args = args)

        # create "alignment" of human fasta and true splice sites (via Intron/Exon state sequence)
        if single_species == "Homo_sapiens":
            write_clw_true_state_seq(stripped_fasta_file_path, out_dir_path = exon_dir, logger = logger)

    if error_species > 0:
        logger.warning(f"Skipped {error_species}/{len(all_species)} species due to liftover errors")

    # gather all usable fasta seqs in a single file, with human sequence first
    input_files = get_input_files_with_human_at_0(from_path = stripped_seqs_dir, logger = logger)

    output_file = os.path.join(exon_dir, "combined.fasta")
    combine_fasta_files(output_file = output_file, input_files = input_files)

    if args.convert_short_lc_to_UC > 0:
        output_file = os.path.join(capitalzed_subs_seqs_dir, "combined.fasta")
        convert_short_lc_to_UC(output_file, input_files, threshold = args.convert_short_lc_to_UC)

    # [ME] also apply above remaining steps to my SequenceRepresentation.Sequence objects, just so the script 
    # parameters work as expected also on my data
    hg_id = None
    for i, sequence in enumerate(sequence_reps):
        if sequence.species == "Homo_sapiens":
            hg_id = i

        sequence.stripSequence(int(args.min_left_exon_len/2))
        sequence.stripSequence(int(args.min_right_exon_len/2), from_start=False)

        if args.convert_short_lc_to_UC > 0 and sequence.sequence is not None:
            new_seq = capitalize_lowercase_subseqs(sequence.sequence, args.convert_short_lc_to_UC)
            sequence.sequence = new_seq

    assert hg_id is not None, "[ERROR] >>> No human sequence found among input species"
    hgseq = sequence_reps.pop(hg_id)
    sequence_reps.insert(0, hgseq)

    # store sequence data
    sequence_reps = [s.toDict() for s in sequence_reps]
    #print("[DEBUG] >>> Sequence Representations:")
    #print(sequence_reps)
    with open(os.path.join(exon_dir, "profile_finding_sequence_data.json"), 'wt') as fh:
        json.dump(sequence_reps, fh, indent=4)

    logger.info(f"Done processing exon {exon_i+1}/{total_i}: " + \
                    f"{exon.seq}_{exon.start_in_genome}_{exon.stop_in_genome}. It took {time.perf_counter() - start}")
    logqueue.put(logger)
