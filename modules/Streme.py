""" Functionality to run STREME for performance evaluation. """

from dataclasses import dataclass
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
from . import ModelDataSet
from . import plotting
from . import SequenceRepresentation as sr
from . import training
from .utils import full_stack



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
    k_min: int # motif width 
    k_max: int # motif width 
    n_best_motifs: int # number of best motifs to report, None to let STREME decide
    streme_exe: str = "streme"
    # adjust this based on whether this is run from AppHub or brain:
    load_streme_script: str = "${HOME}/Software/load_MEME.sh"
    # no need to adjust these:
    _data_file: str = "data.fasta"
    _fahead_mapping_file: str = "sequence_name_mapping.json"
    _streme_outdir: str = "streme_out"

    def __post_init__(self):
        if not os.path.exists(self.working_dir):
            logging.info(f"Creating working directory {self.working_dir}.")
            os.makedirs(self.working_dir)

        self._faheadToSeqidcs: dict[str, dict[str, int|str]] = None # store mapping after run() to restore sequence idcs


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


    def _getStremeOutputSites(self, data: ModelDataSet.ModelDataSet) -> list[Links.MultiLink]:
        """ Load the sites of the found motifs from the STREME output. """
        resultfile = os.path.join(self.working_dir, self._streme_outdir, "sites.tsv")
        if not os.path.exists(resultfile):
            logging.error(f"STREME output file {resultfile} not found.")
            return None

        sites = pd.read_csv(resultfile, sep="\t", comment="#")

        # re-use seqname mapping, fallback to file if not found, otherwise error
        if self._faheadToSeqidcs is None:
            logging.warning(f"Sequence name mapping is None. Loading from file {self._fahead_mapping_file}.")
            with open(os.path.join(self.working_dir, self._fahead_mapping_file), "rt") as fh:
                self._faheadToSeqidcs = json.load(fh)

        # assert that right sequences are referenced        
        for fahead in self._faheadToSeqidcs:
            seq = data.training_data.getSequence(self._faheadToSeqidcs[fahead]['outer_idx'],
                                                 self._faheadToSeqidcs[fahead]['inner_idx'],
                                                 self._faheadToSeqidcs[fahead]['frame_idx'])
            assert seq.id == self._faheadToSeqidcs[fahead]['sequence_id'], \
                f"Sequence ID mismatch: {seq.id} vs. {self._faheadToSeqidcs[fahead]['sequence_id']}"
            
        sitetuples = [] # store sites as tuples (genome_idx, contig_idx, frame_idx, pos)
        for row in sites.itertuples():
            assert re.match(r"STREME-\d+", row.motif_ALT_ID), f"Unexpected motif ID: {row.motif_ALT_ID}"
            assert row.seq_ID in self._faheadToSeqidcs, f"Sequence ID {row.seq_ID} not found in sequence name mapping."
            seq = data.training_data.getSequence(self._faheadToSeqidcs[row.seq_ID]['outer_idx'],
                                                 self._faheadToSeqidcs[row.seq_ID]['inner_idx'],
                                                 self._faheadToSeqidcs[row.seq_ID]['frame_idx'])
            
            if data.training_data.datamode in [ModelDataSet.DataMode.Translated, 
                                               ModelDataSet.DataMode.Translated_noStop]:
                assert row.site_Strand == ".", f"Unexpected strand for AA sequence: {row.site_Strand}"
                frame = self._faheadToSeqidcs[row.seq_ID]['frame_idx']
            else:
                assert row.site_Strand in ["+", "-"], f"Unexpected strand for genomic sequence: {row.site_Strand}"
                frame = 0 if row.site_Strand == "+" else 1
                
            position = row.site_Start-1 # STREME uses 1-based positions

            # dbg msg
            # logging.debug(f"[_getStremeOutputSites] row: {row}")
            # logging.debug(f"[_getStremeOutputSites] {row.seq_ID=} {self._faheadToSeqidcs[row.seq_ID]=}")
            # logging.debug(f"[_getStremeOutputSites] {seq.id=} {len(seq)=} {frame=} {position=}")

            sitetuples.append((self._faheadToSeqidcs[row.seq_ID]['outer_idx'],
                               self._faheadToSeqidcs[row.seq_ID]['inner_idx'],
                               frame, position))
            
        occurrences = ModelDataSet.siteConversionHelper(data, sitetuples, self.k_min)
        links = Links.multiLinksFromOccurrences(occurrences)

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
            

    def run(self, runID, data: ModelDataSet.ModelDataSet, #list[sr.Sequence | sr.TranslatedSequence], genomes: list[sr.Genome], 
            evaluator: training.MultiTrainingEvaluation,
            dryrun: bool = False,
            plot_motifs: bool = False, plot_links: bool = False, 
            plot_linkThreshold: int = 1000, plot_onlyLinkedSeqs: bool = True,
            plot_font = None,
            verbose=False, **kwargs) -> list[Links.Link]:
        """ Run STREME on the given data. 
        Args:
            runID: ID of the current run
            data: ModelDataSet.ModelDataSet to run STREME on
            # data: list of SequenceRepresentation.Sequences/TranslatedSequences to run STREME on
            # genomes: list of SequenceRepresentation.Genomes that contain (at least) the data
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
        mode = 'dna' if data.training_data.datamode == ModelDataSet.DataMode.DNA else 'protein'

        # flatten data, write to fasta with simple identifiers, store mapping of fasta identifiers to sequence indices
        flat_seqs = []
        self._faheadToSeqidcs = {} # map fasta headers to dict of sequence indices in `data`
        for outer_idx, inner_list in enumerate(data.getRawData()):
            for inner_idx, frame_seqs in enumerate(inner_list):
                for frame_idx, seq in enumerate(frame_seqs):
                    i = len(flat_seqs)
                    alias = f"seq{i}"

                    sequence = sr.Sequence(f"spec{i}", "chrN", "+", 0, sequence=seq) # dummy sequence
                    sequence.id = alias # set simple ID
                    flat_seqs.append(sequence)

                    frame = frame_idx if data.training_data.datamode in [ModelDataSet.DataMode.Translated, 
                                                                         ModelDataSet.DataMode.Translated_noStop] else 0
                    self._faheadToSeqidcs[alias] = {
                        'outer_idx': outer_idx,
                        'inner_idx': inner_idx,
                        'frame_idx': frame,
                        'sequence_id': data.training_data.getSequence(outer_idx, inner_idx, frame).id
                    }
                    
        sr.sequenceListToFASTA(flat_seqs, os.path.join(self.working_dir, self._data_file))
        with open(os.path.join(self.working_dir, self._fahead_mapping_file), "wt") as fh:
            json.dump(self._faheadToSeqidcs, fh, indent=2)

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
            # multilinks = self._getStremeOutputSites(data)
            mlinks = self._getStremeOutputSites(data)
            motifs, parser = self._getStremeOutputMotifs()
            motif_meta = {'attr': parser.motif_attributes,
                          'alphabet': parser.alphabet}

            hgGenome = [g for g in data.training_data.getGenomes() if g.species == 'Homo_sapiens']
            assert len(hgGenome) == 1, f"[ERROR] >>> found {len(hgGenome)} human genomes: {hgGenome}"
            hgGenome = hgGenome[0]
            evaluator.add_result(runID, 
                                 motifs=training.MotifWrapper(motifs=motifs, alphabet=parser.alphabet, 
                                                              metadata=motif_meta), 
                                 # links=multilinks, 
                                 links=mlinks,
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
                logging.debug(full_stack())
                return None

        if plot_links:
            try:
                img = plotting.drawGeneLinks(links=mlinks, genomes=data.training_data.getGenomes(),
                                             imname=os.path.join(self.working_dir, self._streme_outdir, "links.png"),
                                             onlyLinkedGenes=plot_onlyLinkedSeqs,
                                             font = plot_font, **kwargs)
                img.close()
            except Exception as e:
                logging.error("[Streme.run] Plotting Links failed.")
                logging.error(f"[Streme.run] Exception:\n{e}")
                logging.debug(full_stack())
                return None
