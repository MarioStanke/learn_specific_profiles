""" Functionality to run STREME for performance evaluation. """

from dataclasses import dataclass, field
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import shutil
import subprocess
from time import time
from xml.parsers import expat

from . import Links
from . import plotting
from . import SequenceRepresentation as sr
from . import sequtils as su
from . import training
from .typecheck import typecheck, typecheck_list



class StremeXMLParser:
    """ Class to handle STREME XML output and parse the result motifs. """
    def __init__(self, filename):
        assert os.path.isfile(filename), f"[ERROR] >>> No file '{filename}' found"
        
        self.file = filename
        self.parser = expat.ParserCreate(encoding = "UTF-8")
        self.motifs = None
        self.motif_attributes = None
        self.current_motif = None
        self.current_pos = None
        self.counting_letters = None
        self.alphabet = []
        
    def getMotifs(self) -> np.array:
        """ Parse the STREME XML output and return the found motifs.
        Returns: np.array of shape (maxwidth, len(alphabet), n_motifs)"""
        def start_handler(name, attrs):
            if name == 'alphabet':
                assert self.counting_letters is None, f"[ERROR] >>> Multiple <alphabet> tags encountered in {self.file}"
                self.counting_letters = True
                assert 'like' in attrs, f"[ERROR] >>> Expected 'like' attribute in 'alphabet' tag: {attrs}"
                if attrs['like'] == 'dna':
                    self.expected_alphabet = ['A', 'C', 'G', 'T']
                elif attrs['like'] == 'protein':
                    self.expected_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R",
                                              "S", "T", "V", "W", "Y"]
                else:
                    self.expected_alphabet = None
                
            if name == 'letter':
                assert self.counting_letters, f"[ERROR] >>> <letter> tag seen outside <alphabet> tags in {self.file}"
                self.alphabet.append(attrs['symbol'])
                
            if name == 'motifs':
                assert self.motifs is None, f"[ERROR] >>> Multiple <motifs> tags enountered in {self.file}"
                self.motifs = []
                self.motif_attributes = []
                
            if name == 'motif':
                assert self.current_motif is None, \
                    f"[ERROR] >>> New <motif> tags enountered while current tag was not closed in {self.file}"
                self.current_motif = np.zeros((int(attrs['width']), len(self.alphabet), 1), dtype=np.float32)
                self.current_pos = 0
                self.motif_attributes.append(attrs)
                
            if name == 'pos':
                assert not self.current_motif is None, \
                    f"[ERROR] >>> New <pos> tag outside <motif> encountered in {self.file}"
                for i, s in enumerate(self.alphabet):
                    if s in attrs:
                        self.current_motif[ self.current_pos, i, 0 ] = float(attrs[s])
                        
                self.current_pos += 1
                
        
        def end_handler(name):
            if name == 'alphabet':
                assert self.counting_letters, f"[ERROR] >>> </alphabet> with no opening tag encountered in {self.file}"
                self.counting_letters = False
                
            if name == 'motifs':
                assert not self.motifs is None, f"[ERROR] >>> </motifs> with no opening tag enountered in {self.file}"
                
            if name == 'motif':
                assert not self.current_motif is None, \
                    f"[ERROR] >>> </motif> with no opening tag encountered in {self.file}"
                assert not self.motifs is None, f"[ERROR] >>> </motifs> with no opening tag enountered in {self.file}"
                
                self.motifs.append(self.current_motif)
                self.current_motif = None
                self.current_pos = None
                
        self.parser.StartElementHandler = start_handler
        self.parser.EndElementHandler = end_handler
        
        with open(self.file, 'rb') as fh:
            self.parser.ParseFile(fh)
            
        maxwidth = max([m.shape[0] for m in self.motifs])
        motifs = np.zeros((maxwidth, len(self.alphabet), len(self.motifs)))
        for i, m in enumerate(self.motifs):
            motifs[0:m.shape[0], :, i] = m[:,:,0]
        
        # strip exceeding columns if applicable (e.g. if the alphabet is larger than expected)
        if self.expected_alphabet is not None:
            if len(self.alphabet) < len(self.expected_alphabet):
                logging.warning(f"Observed alphabet {self.alphabet} does not match expected: {self.expected_alphabet}")
            elif not (self.alphabet[:len(self.expected_alphabet)] == self.expected_alphabet):
                logging.warning(f"Beginning of observed alphabet {self.alphabet} does not match expected: " +\
                                f"{self.expected_alphabet}")
            elif len(self.alphabet) > len(self.expected_alphabet):
                exceeding = len(self.alphabet) - len(self.expected_alphabet)
                if np.all(motifs[:,len(self.expected_alphabet):,:] \
                            == np.zeros((motifs.shape[0], exceeding, motifs.shape[2]))):
                    motifs = motifs[:,:len(self.expected_alphabet),:]
                    self.alphabet = self.expected_alphabet # drop additional characters
                    
        return motifs



@dataclass
class Streme:
    working_dir: str
    k_min: int = 20 # motif width 
    k_max: int = 20 # motif width 
    n_best_motifs: int = 2 # number of best motifs to report, None to let STREME decide
    streme_exe: str = "streme"
    # adjust this based on whether this is run from AppHub or brain:
    load_streme_script: str = "${HOME}/Software/load_MEME.sh"
    # no need to adjust these:
    _data_file: str = "data.fasta"
    _seqname_mapping_file: str = "sequence_name_mapping.json"
    _streme_outdir: str = "streme_out"

    def __post_init__(self):
        if not os.path.exists(self.working_dir):
            logging.info(f"Creating working directory {self.working_dir}.")
            os.makedirs(self.working_dir)

        self._seqname_mapping = None # store mapping after run() to restore sequence IDs


    def _getStremeOutputMotifs(self):
        """ Load the found motifs from the STREME output. Returns an array of shape (maxwidth, len(alphabet), n_motifs)
        and the parser object. """
        resultfile = os.path.join(self.working_dir, self._streme_outdir, "streme.xml")
        if not os.path.exists(resultfile):
            logging.error(f"STREME output file {resultfile} not found.")
            return None, None

        # parse the XML file
        parser = StremeXMLParser(resultfile)
        return parser.getMotifs(), parser



    def _getStremeOutputSites(self, data: list[sr.Sequence | sr.TranslatedSequence]) -> list[Links.MultiLink]:
        """ Load the sites of the found motifs from the STREME output. """
        resultfile = os.path.join(self.working_dir, self._streme_outdir, "sites.tsv")
        if not os.path.exists(resultfile):
            logging.error(f"STREME output file {resultfile} not found.")
            return None

        sites = pd.read_csv(resultfile, sep="\t", comment="#")

        # re-use seqname mapping, fallback to file if not found, otherwise error
        if self._seqname_mapping is None:
            logging.warning(f"Sequence name mapping is None. Loading from file {self._seqname_mapping_file}.")
            with open(os.path.join(self.working_dir, self._seqname_mapping_file), "rt") as fh:
                self._seqname_mapping = json.load(fh)
        
        assert sorted(self._seqname_mapping.values()) == sorted([s.id for s in data]), \
            f"Sequence IDs in sequence name mapping do not match the input data: {self._seqname_mapping.values()} " \
            + f"vs. {[s.id for s in data]}"
        
        # map STREME 'seq_ID' to data sequences
        trueSeqIDs = []
        for sid in sites['seq_ID']:
            tsid = self._seqname_mapping.get(str(sid), None)
            if tsid is None:
                raise ValueError(f"Sequence ID {sid} of type {type(sid)} " \
                                 + f"not found in sequence name mapping {self._seqname_mapping}.")

            trueSeqIDs.append(tsid)

        sites['seq_ID'] = trueSeqIDs # restore actual sequence IDs from mapping
        data_seqdict: dict[str, sr.Sequence | sr.TranslatedSequence] = {s.id: s for s in data}
        
        # generate Links from the sites
        motifToOccs = {}
        for row in sites.itertuples():
            assert re.match(r"STREME-\d+", row.motif_ALT_ID), f"Unexpected motif ID: {row.motif_ALT_ID}"
            if row.seq_ID not in data_seqdict:
                logging.critical(f"Sequence ID {row.seq_ID} not found in input data.")
                continue

            seq: sr.Sequence | sr.TranslatedSequence = data_seqdict[row.seq_ID]
            assert typecheck(seq, ["Sequence", "TranslatedSequence"], die=True)

            # DEBUG: check that row.site_Sequence matches the sequence at the given position!
            siteseq = row.site_Sequence
            a = row.site_Start-1
            b = row.site_End
            assert seq.sequence[a:b] == siteseq, f"Sequence mismatch: {seq.sequence[a:b]} vs. {siteseq} in row {row}"
            # -------------------------------------------------------------------------------

            if typecheck(seq, "TranslatedSequence", die=False, log_warnings=False):
                seq: sr.TranslatedSequence = seq # only for linting
                assert row.site_Strand == ".", f"Unexpected strand for AA sequence: {row.site_Strand}"
                strand = "+" if seq.frame < 3 else "-"
                # convert_six_frame_position returns the position of the first codon base w.r.t. the forward strand for
                #   the codon that translates into the first aa of the sequence. 
                # I.e. for frame 3, aa[0,1,2,...] <-> dna[len(dna)-3, len(dna)-6, len(dna)-9, ...]
                aa_pos = row.site_Start-1 if seq.frame < 3 else row.site_End-1 # to get the first dna pos w.r.t. fwd
                position = su.convert_six_frame_position(aa_pos, seq.frame, seq.genomic_sequence.length, 
                                                         dna_to_aa=False)
                
                # DEBUG: check that the translated sequence at the converted position matches row.site_Sequence!
                siteseq = row.site_Sequence
                gseq = seq.genomic_sequence #genomes[genome_idxs[0]][genome_idxs[1]]
                gsitelen = len(siteseq) * 3
                gsiteseq = gseq.sequence[position:position+gsitelen]
                assert su.sequence_translation(gsiteseq, seq.frame >= 3) == siteseq, f"Sequence mismatch: {su.sequence_translation(gsiteseq, seq.frame >= 3)} vs. {siteseq} in row {row}"
                # -------------------------------------------------------------------------------
                oseq = seq.genomic_sequence

            else: # seq: Sequence
                assert row.site_Strand in ["+", "-"], f"Unexpected strand for genomic sequence: {row.site_Strand}"
                # genome_idxs = genome_seqToIdxs[seq.id]
                strand = row.site_Strand
                position = row.site_Start-1
                # ^^^ checked: STREME uses 1-based positions; if the motif is on the reverse strand, the position still
                #              refers to the top strand (the `site_Sequence` is reverse-complemented, but the 
                #              `site_Start` and `site_End` are not)    
                oseq = seq

            occ = Links.Occurrence(sequence = oseq,
                                   position = position, 
                                   strand = strand,
                                   profileIdx = int(row.motif_ALT_ID.split('-')[1]) - 1) # streme starts counting at 1
            
            if row.motif_ID not in motifToOccs:
                motifToOccs[row.motif_ID] = []

            motifToOccs[row.motif_ID].append(occ)

        # create Links
        links = [
            Links.MultiLink(occs = motifToOccs[motif], span = len(motif.split('-')[1]), singleProfile = True)
                for motif in motifToOccs
        ]

        return links



    def _makeStremeCommand(self, mode: str):
        """ Create STREME command in a make.doc file. `mode` has to be one of 'dna', 'rna', 'protein'. """
        assert mode in ['dna', 'rna', 'protein'], f"Unexpected mode: {mode}"

        nmotifs = f"--nmotifs {self.n_best_motifs}" if self.n_best_motifs is not None else ""
        make_doc = f"""
# load the MEME suite
#module load gcc/13.2.0 openmpi/4.1.1
source {self.load_streme_script}

# settings
input="{self._data_file}"
outdir="{self._streme_outdir}"
alphabet="{mode}" # one of 'dna', 'rna', 'protein'
nmotifs="{nmotifs}" # empty string to let streme decide
minmotifw="{self.k_min}"
maxmotifw="{self.k_max}"
"""
        make_doc += """
# run STREME
optionstr=" \
--p ${input} \
--o ${outdir} \
--${alphabet} \
${nmotifs} \
--minw ${minmotifw} \
--maxw ${maxmotifw}"

echo "Running '"""+self.streme_exe+""" ${optionstr}'"
"""+self.streme_exe+""" ${optionstr}"""

        with open(os.path.join(self.working_dir, "make.doc"), "w") as f:
            f.write(make_doc)

    
    def _plotStremeOutput(self, multilinks: list[Links.MultiLink], genomes: list[sr.Genome], 
                          plot_onlyLinkedSeqs: bool = True,
                          plot_linkThreshold: int = 100,
                          plot_font = None, **kwargs):
        """ Plot the output of STREME. """
        links = []
        for ml in multilinks:
            ls = ml.toLinks(linkThreshold = plot_linkThreshold)
            if ls is not None:
                links.extend(ls)

        return plotting.drawGeneLinks(genomes=genomes, links=links, 
                                      imname=os.path.join(self.working_dir, self._streme_outdir, "links.png"),
                                      onlyLinkedGenes=plot_onlyLinkedSeqs,
                                      font = plot_font, **kwargs)

    
    def run(self, runID, data: list[sr.Sequence | sr.TranslatedSequence], genomes: list[sr.Genome], 
            evaluator: training.MultiTrainingEvaluation,
            dryrun: bool = False,
            plot_motifs: bool = False, plot_links: bool = False, 
            plot_linkThreshold: int = 100, plot_onlyLinkedSeqs: bool = True,
            plot_font = None,
            verbose=False, **kwargs) -> list[Links.Link]:
        """ Run STREME on the given data. 
        Args:
            runID: ID of the current run
            data: list of SequenceRepresentation.Sequences/TranslatedSequences to run STREME on
            genomes: list of SequenceRepresentation.Genomes that contain (at least) the data
            evaluator: training.MultiTrainingEvaluation object to use for performance evaluation over multiple runs
            dryrun: whether to skip the actual STREME execution and subsequent plotting, i.e. do only run preparation
                    and return None
            plot_motifs: whether to plot the motifs
            plot_links: whether to plot the links
            plot_font: optional, font to use for plotting (if None, use default font from this repository)
            verbose: whether to log STREME output
            **kwargs: additional arguments to pass to plotting.drawGeneLinks
        Returns:
            list of Links.MultiLink objects representing the found motif sites in the data
        """
        for s in data:
            typecheck_list(s, ["Sequence", "TranslatedSequence"], die=True)

        seqtype = set([s.classname for s in data])
        assert len(seqtype) == 1, f"Multiple sequence types encountered: {seqtype}"
        seqtype = seqtype.pop()
        mode = 'dna' if seqtype == "Sequence" else 'protein'

        # rename sequences to simple numbers for STREME, write data and name mapping to files, then re-rename
        self._seqname_mapping = {}
        for i, seq in enumerate(data):
            alias = f"seq{i}"
            self._seqname_mapping[alias] = seq.id
            seq.id = alias

        sr.sequenceListToFASTA(data, os.path.join(self.working_dir, self._data_file))
        with open(os.path.join(self.working_dir, self._seqname_mapping_file), "wt") as fh:
            json.dump(self._seqname_mapping, fh, indent=2)

        for seq in data:
            seq._regenerateID()

        # create make.doc
        self._makeStremeCommand(mode)

        if dryrun:
            return None

        # run STREME
        try:
            start = time()
            p = subprocess.run(f"{shutil.which('bash')} make.doc", cwd=self.working_dir, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               shell=True)
            end = time()
            training_time = end-start
            logging.info(f"STREME took {training_time:.2f} seconds.")
            if verbose:
                logging.info(f"STREME output: {p.stdout}")

            # load the sites of the found motifs from the STREME output
            multilinks = self._getStremeOutputSites(data)
            motifs, parser = self._getStremeOutputMotifs()
            motif_meta = {'attr': parser.motif_attributes,
                          'alphabet': parser.alphabet}

            hgGenome = [g for g in genomes if g.species == 'Homo_sapiens']
            assert len(hgGenome) == 1, f"[ERROR] >>> found {len(hgGenome)} human genomes: {hgGenome}"
            hgGenome = hgGenome[0]
            evaluator.add_result(runID, 
                                 motifs=training.MotifWrapper(motifs=motifs, metadata=motif_meta), 
                                 links=multilinks, 
                                 hg_genome=hgGenome,
                                 time=training_time)

        except subprocess.CalledProcessError as e:
            print(f"DEBUG: e: {e}")
            print(f"DEBUG: e.stdout: {e.output}")
            print(f"DEBUG: e.stderr: {e.stderr}")
            logging.error(f"STREME failed with exit code {e.returncode}.")
            logging.error(f"Command: {' '.join(e.cmd)}")
            logging.error(f"Output: {e.output}")
            logging.error(f"Stderr: {e.stderr}")
            return None

        # plot the results
        if plot_motifs:
            try:
                motifs, parser = self._getStremeOutputMotifs()
                mNames = [a['id'] for a in parser.motif_attributes]
                plotting.plotLogo(motifs, alphabet=parser.alphabet, pNames=mNames)
            except Exception as e:
                logging.error("[Streme.run] Plotting motifs failed.")
                logging.error(f"[Streme.run] Exception:\n{e}")
                return None

        if plot_links:
            try:
                _ = self._plotStremeOutput(multilinks, genomes,
                                           plot_linkThreshold=plot_linkThreshold,
                                           plot_onlyLinkedSeqs=plot_onlyLinkedSeqs,
                                           plot_font=plot_font, **kwargs)
            except Exception as e:
                logging.error("[Streme.run] Plotting Links failed.")
                logging.error(f"[Streme.run] Exception:\n{e}")
                return None
