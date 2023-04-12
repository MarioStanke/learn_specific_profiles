#!/usr/bin/env python3
import pandas as pd
import json
import argparse
import os
import time
import re
from Bio import SeqIO
import math

################################################################################
def get_output_dir():
    lengths_config_str = str(args.min_left_neighbour_exon_len)
    lengths_config_str += "_" + str(args.len_of_left_to_be_lifted)
    lengths_config_str += "_" + str(args.min_left_neighbour_intron_len)
    lengths_config_str += "_" + str(args.min_exon_len)
    lengths_config_str += "_" + str(args.len_of_exon_middle_to_be_lifted)
    lengths_config_str += "_" + str(args.min_right_neighbour_intron_len)
    lengths_config_str += "_" + str(args.len_of_right_to_be_lifted)
    lengths_config_str += "_" + str(args.min_right_neighbour_exon_len)

    # dirs
    output_dir = f"{args.path}/out_{'' if not args.n else str(args.n) + 'Exons_'}{args.species.split('/')[-1]}_{lengths_config_str}_{args.hal.split('/')[-1]}"
    if not os.path.exists(output_dir):
        os.system(f"mkdir -p {output_dir}")
    return output_dir
################################################################################
def load_hg38_refseq_bed():
    start = time.perf_counter()
    print("started load_hg38_refseq_bed()")
    hg38_refseq_bed = pd.read_csv(args.hg38, delimiter = "\t", header = None)
    hg38_refseq_bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    hg38_refseq_bed["blockSizes"] = hg38_refseq_bed["blockSizes"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    hg38_refseq_bed["blockStarts"] = hg38_refseq_bed["blockStarts"].apply(lambda s: [int(a) for a in s[:-1].split(",")])
    print("finished load_hg38_refseq_bed(). It took:", time.perf_counter() - start)
    return hg38_refseq_bed
################################################################################
def get_all_internal_exons(hg38_refseq_bed):
    start = time.perf_counter()
    print("started get_all_internal_exons()")
    internal_exons = {} # key is chromosom, start and stop of exon in genome, value is list of rows that mapped to this exon range
    for index, row in hg38_refseq_bed.iterrows():
        if args.n and index > args.n * 100: # cant just break at args.n since, exons are filtered in an additional step. So i have to build some more here, such that after filtering sufficiently many remain.
            break
        if row["blockCount"] < 3:
            continue

        # print(row)
        for exon_id, (exon_len, exon_start) in enumerate(zip(row["blockSizes"][1:-1], row["blockStarts"][1:-1])):
            assert row["chromStart"] <= row["chromEnd"], 'row["chromStart"] <= row["chromEnd"]'
            assert row["thickStart"] <= row["thickEnd"], 'row["thickStart"] <= row["thickEnd"]'
            chromosom = row["chrom"]
            exon_start_in_genome = row["chromStart"] + exon_start
            exon_end_in_genome = row["chromStart"] + exon_start + exon_len # the end id is not included
            key = (chromosom, exon_start_in_genome, exon_end_in_genome)
            if key in internal_exons:
                internal_exons[key].append(row)
            else:
                internal_exons[key] = [row]
    if args.v:
        print("since the same exon occures in multiple genes, which may just be spliced differently")
        for i, key in enumerate(sorted(internal_exons.keys())):
            if len(internal_exons[key]) > 3:
                print(f"key is exon = {key}")
                print("value is list of df_rows:")
                for df_row in internal_exons[key]:
                    print(df_row)
                break
    print("finished get_all_internal_exons(). It took:", time.perf_counter() - start)
    return internal_exons
################################################################################
def filter_and_choose_exon_neighbours(all_internal_exons):
    '''
    bc of alternative spliced genes, an exon might have multiple exons neigbhours to choose one.

    '''
    start = time.perf_counter()
    print("started filter_and_choose_exon_neighbours()")
    filtered_internal_exons = []
    for i, key in enumerate(all_internal_exons.keys()):
        if args.n and len(filtered_internal_exons) >= args.n:
            break
        for row in all_internal_exons[key]:
            exon_start_in_gene = key[1] - row["chromStart"]
            exon_id = row["blockStarts"].index(exon_start_in_gene)
            assert row["blockSizes"][exon_id] == key[2] - key[1], "calculated id exon len is not same as stop - start in genome"
            if row["blockSizes"][exon_id] < args.min_exon_len:
                continue
            if row["blockSizes"][exon_id - 1] < args.min_left_neighbour_exon_len:
                continue
            left_intron_len = row["blockStarts"][exon_id] - row["blockStarts"][exon_id-1] - row["blockSizes"][exon_id - 1]
            if left_intron_len < args.min_left_neighbour_intron_len:
                continue
            right_intron_len = row["blockStarts"][exon_id + 1] - row["blockStarts"][exon_id] - row["blockSizes"][exon_id]
            if right_intron_len < args.min_right_neighbour_intron_len:
                continue
            if row["blockSizes"][exon_id + 1] < args.min_right_neighbour_exon_len:
                continue

            # getting coordinates from rightmost end of left exon that will be lifted to the other genomes
            # getting coordinates from leftmost end of right exon that will be lifted to the other genome
            left_lift_start = key[1] - left_intron_len - args.len_of_left_to_be_lifted
            left_lift_end = left_lift_start + args.len_of_left_to_be_lifted
            right_lift_start = key[2] + right_intron_len
            right_lift_end = right_lift_start + args.len_of_right_to_be_lifted

            di = {"seq" : key[0], \
                  "start_in_genome" : key[1], \
                  "stop_in_genome" : key[2], \
                  "exon_id" : exon_id, \
                  "left_lift_start": left_lift_start, \
                  "left_lift_end" : left_lift_end, \
                  "right_lift_start" : right_lift_start, \
                  "right_lift_end" : right_lift_end, \
                  "left_intron_len" : left_intron_len, \
                  "right_intron_len" : right_intron_len, \
                  "row" : dict(row)}
            filtered_internal_exons.append(di)

            # for an internal exon the neigbhours are choosen
            # (simply the first occurance is selected)
            # so other occurances of this exon in alternatively spliced genes (= rows in hg38.bed)
            # are discarded
            break

    if args.v:
        for i in range(3):
            print()
            for key in sorted(filtered_internal_exons[i].keys()):
                print(key, filtered_internal_exons[i][key])
    print("finished filter_and_choose_exon_neighbours(). It took:", time.perf_counter() - start)
    return filtered_internal_exons
################################################################################
def write_filtered_internal_exons(filtered_internal_exons):
    start = time.perf_counter()
    print("started write_filtered_internal_exons()")
    with open(json_path, "w") as json_out:
        json.dump(filtered_internal_exons, json_out)
    print("finished write_filtered_internal_exons(). It took:", time.perf_counter() - start)
################################################################################
def get_to_be_lifted_exons(hg38_refseq_bed):
    if os.path.exists(json_path) and not args.overwrite:
        print(f"the file {json_path} exists, so it isnt calculated again")
        start = time.perf_counter()
        print("started json.load(filtered_internal_exons.json)")
        with open(json_path) as file:
            filtered_internal_exons = json.load(file)
        print("finished json.load(filtered_internal_exons.json). It took:", time.perf_counter() - start)
    else:
        all_internal_exons = get_all_internal_exons(hg38_refseq_bed)
        filtered_internal_exons = filter_and_choose_exon_neighbours(all_internal_exons)
        write_filtered_internal_exons(filtered_internal_exons)
    if args.n:
        filtered_internal_exons = filtered_internal_exons[:args.n]
    return filtered_internal_exons
################################################################################
################################################################################
################################################################################
def create_bed_file_to_be_lifted(exon = None, out_path = None):
    # seq     start           stop            name    score   strand
    # chr1    67093589        67093604        left    0       -
    with open(out_path, "w") as bed_file:
        def add_bed_line(seq = exon["seq"], start = None, stop = None, name = None, score = "0", strand = exon["row"]["strand"]):
            bed_file.write(seq + "\t")
            bed_file.write(start + "\t")
            bed_file.write(stop + "\t")
            bed_file.write(name + "\t")
            bed_file.write(score + "\t")
            bed_file.write(strand + "\n")

        base_name = f"exon_{exon['start_in_genome']}_{exon['stop_in_genome']}_{exon['exon_id']}"

        # left and right neighbouring exon
        for left_or_right in ["left","right"]:
            add_bed_line(start = str(exon[f"{left_or_right}_lift_start"]), \
            stop = str(exon[f"{left_or_right}_lift_end"]), \
            name = f"{base_name}_{left_or_right}")

        # middle of exon
        left_middle = (exon["stop_in_genome"] + exon['start_in_genome'] - args.len_of_exon_middle_to_be_lifted)//2
        right_middle = left_middle + args.len_of_exon_middle_to_be_lifted # this index does not part of the area to be lifted
        add_bed_line(start = str(left_middle), \
        stop = str(right_middle), \
        name = f"{base_name}_middle")

        # start and stop of exon

        # add_bed_line(start = str(exon["start_in_genome"]), \
        #              stop = str(exon["start_in_genome"] + args.len_of_exon_middle_to_be_lifted), \
        #              name = f"{base_name}_exonstart")
        # add_bed_line(start = str(exon["stop_in_genome"] -  args.len_of_exon_middle_to_be_lifted), \
        #              stop = str(exon["stop_in_genome"]), \
        #              name = f"{base_name}_exonend")
################################################################################
def get_new_or_old_species_bed(human_exon_to_be_lifted_path = None, species_name = None, out_dir = None):
    ''' either creates new lifted over bed file and returns path to it
        or it returns file to existing bed file
    '''
    bed_file_path = f"{out_dir}/{species_name}.bed"
    if not args.use_old_bed:
        command = f"time halLiftover {args.hal} Homo_sapiens {human_exon_to_be_lifted_path} {species_name} {bed_file_path}"
        print("running:", command)
        os.system(command)
        return 1
    else:
        bed_files = [f for f in os.listdir(out_dir) if f.endswith(".bed")]
        for bed_file in bed_files:
            # if bed_file.startswith(single_species):
            #     return f"{out_dir}/{bed_file}"
            if bed_file == f"{species_name}.bed":
                return 1
################################################################################
def extract_info_and_check_bed_file(bed_dir = None, species_name = None, extra_seq_data = None, extra_exon_data = None):
    bed_file_path = f"{bed_dir}/{species_name}.bed"

    if os.path.getsize(bed_file_path) == 0:
        os.system(f"mv {bed_file_path} {bed_dir}/{species_name}_errorcode_empty.bed")
        return False

    species_bed = pd.read_csv(bed_file_path, delimiter = "\t", header = None)
    species_bed.columns = ["seq", "start", "stop", "name", "score", "strand"]
    if len(species_bed.index) != 3 and args.discard_multiple_bed_hits:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_more_than_3_lines.bed")
        return False
    l_m_r = {}
    for index, row in species_bed.iterrows():
        x = re.search(r"\d+_\d+_\d+_(.+)",row["name"])
        try:
            if (y := x.group(1)) in l_m_r: # y = left, right or middle
                len_of_previous_bed_hit = l_m_r[y]["stop"] - l_m_r[y]["start"]
                len_of_current_bed_hit = row["stop"] - row["start"]
                if len_of_current_bed_hit > len_of_previous_bed_hit:
                    l_m_r[y] = row
            else:
                l_m_r[y] = row
        except:
            print("l_m_r[x.group(1)] didnt work")
            print("row['name']", row["name"])
            exit()

    if len(l_m_r) != 3:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_not_all_l_m_r.bed")
        return False
    if l_m_r["left"]["strand"] != l_m_r["right"]["strand"] or l_m_r["left"]["strand"] != l_m_r["middle"]["strand"]:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_unequal_strands.bed")
        return False
    if l_m_r["left"]["seq"] != l_m_r["left"]["seq"] or l_m_r["left"]["seq"] != l_m_r["middle"]["seq"]:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_unequal_seqs.bed")
        return False

    # if strand is opposite to human, left and right swap
    if extra_exon_data["human_strand"] == l_m_r["left"]["strand"]:
        extra_seq_data["seq_start_in_genome"] = l_m_r["left"]["stop"]
        extra_seq_data["seq_stop_in_genome"] = l_m_r["right"]["start"]
    else:
        # i think        [left_start, left_stop] ... [middle_start, middle_stop] ... [right_start, right_stop]
        # gets mapped to [right_start, right_stop] ... [middle_start, middle_stop] ... [left_start, left_stop]
        extra_seq_data["seq_start_in_genome"]  = l_m_r["right"]["stop"]
        extra_seq_data["seq_stop_in_genome"]  = l_m_r["left"]["start"]
    extra_seq_data["middle_of_exon_start"] = l_m_r["middle"]["start"]
    extra_seq_data["middle_of_exon_stop"] = l_m_r["middle"]["stop"]

    if extra_seq_data["seq_start_in_genome"] >= extra_seq_data["middle_of_exon_start"]:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_left_greater_middle.bed")
        return False
    if extra_seq_data["middle_of_exon_stop"] >= extra_seq_data["seq_stop_in_genome"]:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_right_less_middle.bed")
        return False

    extra_seq_data["on_reverse_strand"] = l_m_r["left"]["strand"] == "-"
    extra_seq_data["seq_name"] = l_m_r['left']['seq']
    extra_seq_data["len_of_seq_substring_in_single_species"] = extra_seq_data["seq_stop_in_genome"] - extra_seq_data["seq_start_in_genome"]

    threshold = 1
    l1 = extra_seq_data["len_of_seq_substring_in_single_species"]
    l2 = extra_exon_data["len_of_seq_substring_in_human"]
    if abs(math.log10(l1) - math.log10(l2)) >= threshold:
        os.system(f"mv {bed_dir}/{species_name}.bed {bed_dir}/{species_name}_errorcode_lengths_differ_substantially.bed")
        return False

    return True
################################################################################
def write_extra_data_to_fasta_description_and_reverse_complement(fa_path = None, extra_seq_data = None, exon = None):
    for i, record in enumerate(SeqIO.parse(fa_path, "fasta")):
        assert i == 0, f"found more than one seq in fasta file {fa_path}"

        # write coordinates in genome to seq description
        with open(fa_path, "w") as out_file:
            assert len(record.seq) == extra_seq_data["seq_stop_in_genome"] - extra_seq_data["seq_start_in_genome"], "non stripped: actual seq len and calculated coordinate len differ"

            # if exon is on - strand
            # extracetd fasta is from + strand
            # TAA -> gets converted to TTA
            # these coords are from bed file and hg38:
            # 1 start in genome +     2 exon start +      3 exon stop +      4 stop in genome +
            # these are adjusted to the reversed and complemented fasta
            # 4 start in genomce cd

            description = {"seq_start_in_genome_+_strand" : extra_seq_data["seq_start_in_genome"], \
                           "seq_stop_in_genome_+_strand" : extra_seq_data["seq_stop_in_genome"], \
                           "exon_start_in_human_genome_+_strand": exon['start_in_genome'], \
                           "exon_stop_in_human_genome_+_strand" : exon['stop_in_genome'], \
                           "seq_start_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if not extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "seq_stop_in_genome_cd_strand" : extra_seq_data["seq_start_in_genome"] if extra_seq_data["on_reverse_strand"] else extra_seq_data["seq_stop_in_genome"], \
                           "exon_start_in_human_genome_cd_strand" : exon["start_in_genome"] if not extra_seq_data["on_reverse_strand"] else exon["stop_in_genome"], \
                           "exon_stop_in_human_genome_cd_strand" : exon["stop_in_genome"] if not extra_seq_data["on_reverse_strand"] else exon["start_in_genome"]}
            record.description = json.dumps(description)
            if extra_seq_data["on_reverse_strand"]:
                reverse_seq = record.seq.reverse_complement()
                record.seq = reverse_seq
            SeqIO.write(record, out_file, "fasta")
################################################################################
def run_hal_2_fasta(species_name = None, start = None, len = None, seq = None, outpath = None):
    command = f"time hal2fasta {args.hal} {species_name} --start {start} --length {len} --sequence {seq} --ucscSequenceNames --outFaPath {outpath}"
    print("running:", command)
    os.system(command)
    os.system(f"head {outpath}")
################################################################################
# TODO: can I maybe use the old fasta description such that i dont have to pass extra_seq_data
def strip_seqs(fasta_file = None, exon = None, out_path = None, extra_seq_data = None):
    if os.path.exists(fasta_file):
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            with open(out_path, "w") as stripped_seq_file:
                left_strip_len = int(args.min_left_neighbour_exon_len /2)
                right_strip_len = int(args.min_right_neighbour_exon_len/2)
                record.seq = record.seq[left_strip_len : - right_strip_len]
                seq_start_in_genome = extra_seq_data["seq_start_in_genome"] + left_strip_len
                seq_stop_in_genome = seq_start_in_genome + extra_seq_data["len_of_seq_substring_in_single_species"] - left_strip_len - right_strip_len
                assert seq_stop_in_genome - seq_start_in_genome == len(record.seq), "stripped: actual seq len and calculated coordinate len differ"
                description = {"seq_start_in_genome_+_strand" : seq_start_in_genome, \
                               "seq_stop_in_genome_+_strand" : seq_stop_in_genome, \
                               "exon_start_in_human_genome_+_strand": exon['start_in_genome'], \
                               "exon_stop_in_human_genome_+_strand" : exon['stop_in_genome'], \
                               "seq_start_in_genome_cd_strand" : seq_start_in_genome if not extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "seq_stop_in_genome_cd_strand" : seq_start_in_genome if extra_seq_data["on_reverse_strand"] else seq_stop_in_genome, \
                               "exon_start_in_human_genome_cd_strand" : exon["start_in_genome"] if not extra_seq_data["on_reverse_strand"] else exon["stop_in_genome"], \
                               "exon_stop_in_human_genome_cd_strand" : exon["stop_in_genome"] if not extra_seq_data["on_reverse_strand"] else exon["start_in_genome"]}
                record.description = json.dumps(description)
                SeqIO.write(record, stripped_seq_file, "fasta")
################################################################################
def convert_short_acgt_to_ACGT(outpath, input_files, threshold):
    def capitalize_lowercase_subseqs(seq, threshold_local):
        pattern = f"(?<![a-z])([a-z]{{1,{threshold_local}}})(?![a-z])"
        def repl(match):
            return match.group(1).upper()
        result = re.sub(pattern, repl, seq)
        return str(result)
    with open(outpath, "w") as output_handle:
        for input_file in input_files:
            with open(outpath, "a") as output_handle, open(input_file, "r") as in_file_handle:
                for i, record in enumerate(SeqIO.parse(in_file_handle, "fasta")):
                    assert i == 0, f"convert_short_acgt_to_ACGT found more than one seq in fasta file {input_file}"
                    new_seq = capitalize_lowercase_subseqs(str(record.seq), threshold)
                    output_handle.write(f">{record.id} {record.description}\n")
                    output_handle.write(f"{new_seq}\n")
                    # new_record = record.__class__(seq = "ACGT", id="homo", name="name", description="description")
                    # SeqIO.write(new_record, output_handle, "fasta")
                    # this produced
                    # File "/usr/lib/python3/dist-packages/Bio/File.py", line 72, in as_handle
                    # with open(handleish, mode, **kwargs) as fp:


################################################################################
def get_input_files_with_human_at_0(from_path = None):
    input_files = [f"{from_path}/{f}" for f in os.listdir(from_path) if f.endswith(".fa")]
    input_files = sorted(input_files, key = lambda x: 0 if re.search("Homo_sapiens", x) else 1)
    assert re.search("Homo_sapiens", input_files[0]), "homo sapiens not in first pos of combined.fasta"
    return input_files
################################################################################
def combine_fasta_files(output_file = None, input_files = None):
    with open(output_file, "w") as out:
        for input_file in input_files:
            for seq_record in SeqIO.parse(input_file, "fasta"):
                SeqIO.write(seq_record, out, "fasta")
    out.close()
################################################################################
def create_exon_data_sets(filtered_internal_exons):
    def get_all_species():
        with open(args.species, "r") as species_file:
            species = species_file.readlines()
        return [s.strip() for s in species]
    all_species = get_all_species()

    for exon in filtered_internal_exons:
        exon_dir = f"{output_dir}/exon_{exon['seq']}_{exon['start_in_genome']}_{exon['stop_in_genome']}"
        bed_output_dir = f"{exon_dir}/species_bed"
        seqs_dir = f"{exon_dir}/species_seqs"
        non_stripped_seqs_dir = f"{seqs_dir}/non_stripped"
        stripped_seqs_dir = f"{seqs_dir}/stripped"
        capitalzed_subs_seqs_dir = f"{exon_dir}/combined_fast_capitalized_{args.convert_short_acgt_to_ACGT}"
        extra_exon_data = {}

        for d in [exon_dir, bed_output_dir, seqs_dir, non_stripped_seqs_dir, stripped_seqs_dir, capitalzed_subs_seqs_dir]:
            if not os.path.exists(d):
                os.system(f"mkdir -p {d}")

        human_exon_to_be_lifted_path = f"{exon_dir}/human_exons.bed"

        extra_exon_data["len_of_seq_substring_in_human"] = exon["right_lift_start"] - exon["left_lift_end"]
        extra_exon_data["human_strand"] = exon["row"]["strand"]

        create_bed_file_to_be_lifted(exon = exon, out_path = human_exon_to_be_lifted_path)

        for single_species in all_species:
            extra_seq_data = {}

            if not get_new_or_old_species_bed(
                human_exon_to_be_lifted_path = human_exon_to_be_lifted_path,
                species_name = single_species,
                out_dir = bed_output_dir
            ):
                continue

            if not extract_info_and_check_bed_file(
                bed_dir = bed_output_dir,
                species_name = single_species,
                extra_seq_data = extra_seq_data,
                extra_exon_data = extra_exon_data
            ):
                continue


            # getting the seq, from human: [left exon    [litfed]] [intron] [exon] [intron] [[lifted]right exon]
            # the corresponding seq of [intron] [exon] [intron] in other species
            out_fa_path = f"{non_stripped_seqs_dir}/{single_species}.fa"
            if not args.use_old_fasta:
                run_hal_2_fasta(species_name = single_species, \
                                start = extra_seq_data["seq_start_in_genome"], \
                                len = extra_seq_data["len_of_seq_substring_in_single_species"], \
                                seq = extra_seq_data["seq_name"], \
                                outpath = out_fa_path)

                write_extra_data_to_fasta_description_and_reverse_complement(fa_path = out_fa_path, \
                                                      extra_seq_data = extra_seq_data,
                                                      exon = exon)

            stripped_fasta_file_path = re.sub("non_stripped","stripped", out_fa_path)
            strip_seqs(fasta_file = out_fa_path, \
                       exon = exon, \
                       out_path = stripped_fasta_file_path, \
                       extra_seq_data = extra_seq_data)

            # create alignment of fasta and true splice sites
            if single_species == "Homo_sapiens":
                from Viterbi import fasta_true_state_seq_and_optional_viterbi_guess_alignment
                fasta_true_state_seq_and_optional_viterbi_guess_alignment(stripped_fasta_file_path, out_dir_path = exon_dir)

        # gather all usable fasta seqs in a single file
        input_files = get_input_files_with_human_at_0(from_path = stripped_seqs_dir)

        output_file =f"{exon_dir}/combined.fasta"
        combine_fasta_files(output_file = output_file, input_files = input_files )

        output_file = f"{capitalzed_subs_seqs_dir}/combined.fasta"
        if args.convert_short_acgt_to_ACGT > 0:
            convert_short_acgt_to_ACGT(output_file, input_files, threshold = args.convert_short_acgt_to_ACGT)

def make_stats_table():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(columns = ["path", "exon", "exon_len", "human_seq_len", \
                                 "exon_len_to_human_len_ratio", "median_len", \
                                 "exon_len_to_median_len_ratio","average_len", \
                                 "exon_len_to_average_len", "num_seqs", "ambiguous"])
    dir = get_output_dir() if args.hal else args.stats_table
    for exon in os.listdir(dir):
        exon_dir = os.path.join(dir, exon)
        if os.path.isdir(exon_dir):
            exon_coords = list(map(int, exon.split("_")[2:]))
            exon_len = exon_coords[1] - exon_coords[0]
            lens = []
            for record in SeqIO.parse(f"{exon_dir}/combined.fasta","fasta"):
                lens.append(len(record.seq))
                if record.id.startswith("Homo_sapiens"):
                    human_len = len(record.seq)

            median_len =  np.median(lens)
            average_len = np.average(lens)

            if os.path.exists(f"{exon_dir}/true_alignment_exon_contains_ambiguous_bases.clw"):
                ambiguous = 1
            elif os.path.exists(f"{exon_dir}/true_alignment.clw"):
                ambiguous = -1
            else:
                ambiguous = 0

            new_row_dict = {"path" : exon_dir, \
                            "exon" : exon, \
                            "exon_len" : exon_len, \
                            "human_seq_len" : human_len, \
                            "exon_len_to_human_len_ratio" : exon_len/human_len, \
                            "median_len" :median_len,\
                            "exon_len_to_median_len_ratio" : exon_len/median_len, \
                            "average_len" : average_len, \
                            "exon_len_to_average_len" : exon_len/average_len, \
                            "num_seqs" : len(lens), \
                            "ambiguous" : ambiguous}

            df.loc[len(df)] = new_row_dict
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    df.to_csv(f'{dir}/stats_table.csv', index=True, header=True, line_terminator='\n', sep=";")
    return df


# time halLiftover /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Homo_sapiens human_exon_to_be_lifted.bed Solenodon_paradoxus Solenodon_paradoxus.bed
# time hal2fasta /nas-hs/projs/CGP200/data/msa/241-mammalian-2020v2.hal Macaca_mulatta --start 66848804 --sequence CM002977.3 --length 15 --ucscSequenceNames > maxaxa_exon_left_seq.fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='creating a dataset of exons from hg38.bed and .hal alignment. Only exons which are between introns')
    parser.add_argument('--hg38', help = 'path to hg38-refseq.bed')
    parser.add_argument('--hal', help = 'path to the .hal file')
    parser.add_argument('--species', help = 'path to species file, which are the target of liftover from human')
    parser.add_argument('--min_left_neighbour_exon_len', type = int, default = 20, help = 'min_left_neighbour_exon_len')
    parser.add_argument('--min_left_neighbour_intron_len', type = int, default = 20, help = 'min_left_neighbour_intron_len')
    parser.add_argument('--min_right_neighbour_exon_len', type = int, default = 20, help = 'min_right_neighbour_exon_len')
    parser.add_argument('--min_right_neighbour_intron_len', type = int, default = 20, help = 'min_right_neighbour_intron_len')
    parser.add_argument('--min_exon_len', type = int, default = 50, help = 'min_exon_len')
    parser.add_argument('--len_of_exon_middle_to_be_lifted', type = int, default = 15, help = 'the middle of the exon is also lifted, to check whether it is between left and right if target .bed')
    parser.add_argument('--len_of_left_to_be_lifted', type = int, default = 15, help = 'len_of_left_to_be_lifted')
    parser.add_argument('--len_of_right_to_be_lifted', type = int, default = 15, help = 'len_of_right_to_be_lifted')
    parser.add_argument('--path', default = ".", help = 'working directory')
    parser.add_argument('-n', type = int, help = 'limit the number of exons to n')
    parser.add_argument('-v', action = 'store_true', help = 'verbose')
    parser.add_argument('--use_old_bed', action = 'store_true', help = 'use the old bed files and dont calculate new ones')
    parser.add_argument('--use_old_fasta', action = 'store_true', help = 'use the old fasta files and dont calculate new ones')
    parser.add_argument('--discard_multiple_bed_hits', action = 'store_true', help = 'sometimes, halLiftover maps a single coordinate to 2 or more, if this flag is passed, the species is discarded, otherwise the largest of the hits is selected')
    parser.add_argument('--stats_table', nargs = '?', const = True, help ='instead of getting all the exon data, get stats table of existing data. Specified path, or pass hg38, hal and species and same n')
    parser.add_argument('--convert_short_acgt_to_ACGT', type = int, default = 0, help = 'convert shorter than --convert_short_acgt_to_ACGT')
    args = parser.parse_args()

    assert args.len_of_left_to_be_lifted < args.min_left_neighbour_exon_len, "len_of_left_to_be_lifted > min_left_neighbour_exon_len"
    assert args.len_of_right_to_be_lifted < args.min_right_neighbour_exon_len, "len_of_right_to_be_lifted > min_right_neighbour_exon_len"
    assert args.len_of_exon_middle_to_be_lifted < args.min_exon_len, "len_of_exon_middle_to_be_lifted > min_exon_len"
    # TODO im using the above also for the start and end of th middle exon, not only the middle of the middle/current exon

    if not args.stats_table:
        assert args.hg38 and args.hal and args.species, "you must pass path to hg38, hal and species.lst"
        output_dir = get_output_dir()

        # files
        json_path = f"{output_dir}/filtered_internal_exons.json"

        args.overwrite = False
        if os.path.exists(json_path):
            print("There exists a file with previously exported filtered_internal_exons with same config")
            print("do you want to overwrite it? [y/n] ", end = "")
            while True:
                x = input().strip()
                if x == "y":
                    args.overwrite = True
                    break
                elif x == "n":
                    args.overwrite = False
                    break
                else:
                    print("your answer must be either y or n")

        hg38_refseq_bed = load_hg38_refseq_bed()
        filtered_internal_exons = get_to_be_lifted_exons(hg38_refseq_bed)
        create_exon_data_sets(filtered_internal_exons)
        make_stats_table()
    else:
        if not os.path.isdir(args.stats_table):
            assert args.hg38 and args.hal and args.species, "you must pass path to hg38, hal and species.lst or path to the dir of which the stats table should get created"
        make_stats_table()
