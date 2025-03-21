from dataclasses import dataclass, field
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from time import time

from . import Links
from . import model
from . import ModelDataSet
from . import plotting
from . import ProfileFindingSetup as setup
from . import SequenceRepresentation as sr
from .typecheck import typecheck
from .utils import full_stack


@dataclass
class MotifWrapper():
    """ Bascially store the motifs np.ndarray together with any metadata. TODO: do this properly """
    motifs: np.ndarray
    alphabet: list[str]
    metadata: str = None # type: ignore # can be anything for now, just make sure it is JSON-serializable

    def __post__init__(self):
        assert len(self.motifs.shape) == 3, f"Expected 3D array, got {self.motifs.shape}."
        assert len(self.alphabet) == self.motifs.shape[1], \
            f"Alphabet of length {len(self.alphabet)} does not match motif shape {self.motifs.shape}."

    def toDict(self) -> dict:
        # catch tensors, try to convert them to numpy first
        if callable(getattr(self.motifs, 'numpy', None)):
            try:
                motifs = self.motifs.numpy() # type: ignore
            except Exception as e:
                logging.error(f"[MotifWrapper.toDict] Error converting to numpy.")
                logging.error(f"[MotifWrapper.toDict] Exception:\n{e}")
                logging.info("[MotifWrapper.toDict] Using motif data as is")
                motifs = self.motifs
        else:
            logging.debug(f"[MotifWrapper.toDict] No 'numpy' method found, using motif data as is.")
            motifs = self.motifs

        # do .tolist() stuff manually for any shape
        def recursive_to_list(arr, dtype=float):
            if len(arr.shape) == 0:
                return None
            elif len(arr.shape) == 1:
                return [dtype(x) for x in arr]
            else:
                return [recursive_to_list(a, dtype) for a in arr]
            
        try:
            rmotifs = recursive_to_list(motifs, dtype=float)
            if not (np.array(rmotifs) == motifs).all():
                raise RuntimeError("[MotifWrapper.toDict] Recursive conversion to list failed.")
            motifs = rmotifs

        except Exception as e:
            logging.error(f"[MotifWrapper.toDict] Error converting motifs to float recursively.")
            logging.error(f"[MotifWrapper.toDict] Exception:\n{e}")
            logging.info("[MotifWrapper.toDict] Using .tolist() method")
            motifs = self.motifs.tolist()

        logging.debug(f"[MotifWrapper.toDict] Checking if motifs can be json-dumped")
        _ = json.dumps(motifs) # dies if not possible

        logging.debug(f"[MotifWrapper.toDict] Checking if alphabet can be json-dumped")
        _ = json.dumps(self.alphabet) # dies if not possible

        try:
            _ = json.dumps(self.metadata)
            return {
                'motifs': motifs,
                'alphabet': self.alphabet,
                'metadata': self.metadata
            }
        except Exception as e:
            logging.error(f"[MotifWrapper.toDict] Error converting metadata to JSON.")
            logging.error(f"[MotifWrapper.toDict] Exception:\n{e}")
            logging.info("[MotifWrapper.toDict] Not returning metadata")
            return {
                'motifs': motifs,
                'alphabet': self.alphabet,
                'metadata': None
            }


def loadMotifWrapperFromDict(d: dict) -> MotifWrapper:
    return MotifWrapper(np.asarray(d['motifs']), d['alphabet'], d['metadata'])



@dataclass 
class TrainingEvaluation():
    """ Class to store results of a single training evaluation. """
    runID: str # anything that lets us identify the run later
    motifs: MotifWrapper # motifs found in the training run
    links: list[Links.MultiLink]
    nexonsHit: int
    nhumanExons: int
    nhumanOccs: int # number of occurrences in human
    nhumanOccsThatHit: int # number of occurrences in human that hit an exon
    nlinksThatHit: int
    trainingTime: float # time in seconds, TODO: put that in a history class or sth, does not really fit here

    def __post_init__(self):
        self.nlinks = Links.nLinks(self.links) # type: ignore

    def toDict(self):
        """ Returns a JSON-writable representation of the object. """
        d = {
            'runID': self.runID,
            'nexonsHit': int(self.nexonsHit),
            'nhumanExons': int(self.nhumanExons),
            'nhumanOccs': int(self.nhumanOccs),
            'nhumanOccsThatHit': int(self.nhumanOccsThatHit),
            'nlinks': int(self.nlinks),
            'nlinksThatHit': int(self.nlinksThatHit),
            'trainingTime': float(self.trainingTime),
            'links': [l.toDict() for l in self.links],
            'motifs': self.motifs.toDict()
        }

        # for easier debugging
        for key in d:
            try:
                _ = json.dumps(d[key])
            except Exception as e:
                logging.error(f"[TrainingEvaluation.toDict] >>> Error converting {key} to JSON, dump will fail.")
                logging.error(f"[TrainingEvaluation.toDict] >>> Exception:\n{e}")

        return d
    
    def toMemeTxt(self, file: Path):
        """ Write the motifs to a text file in MEME format. """
        if set(self.motifs.alphabet) == set('ACGT'):
            alphabet = 'ACGT'
        elif set(self.motifs.alphabet) == set('ACGU'):
            alphabet = 'ACGU'
        elif set(self.motifs.alphabet) == set('ACDEFGHIKLMNPQRSTVWY'):
            alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        # elif set(self.motifs.alphabet[:-1]) == set('ACDEFGHIKLMNPQRSTVWY'):
        #     alphabet = 'ACDEFGHIKLMNPQRSTVWY'
            # todo: remove stop codon from motifs, otherwise not possible to write to MEME format
        else:
            raise ValueError(f"[training.TrainingEvaluation.toMemeTxt] Unknown alphabet: {self.motifs.alphabet}")
        
        # make sure the motif rows are in the same order as the alphabet
        if [c for c in alphabet] != self.motifs.alphabet:
            alphabet_map = [self.motifs.alphabet.index(c) for c in alphabet]
            motifs = np.zeros(self.motifs.motifs.shape) # (k, alphabet_size, U_report)
            assert len(motifs.shape) == 3, \
                f"[training.TrainingEvaluation.toMemeTxt] Expected 3D array, got {motifs.shape}."
            for u in range(motifs.shape[2]):
                for i, c in enumerate(alphabet_map):
                    motifs[:, i, u] = self.motifs.motifs[:, c, u]
        else:
            motifs = self.motifs.motifs

        with open(file, 'wt') as fh:
            fh.write(f"MEME version 5\n\n")
            fh.write(f"ALPHABET= {alphabet}\n\n")
            for u in range(motifs.shape[2]):
                motif = motifs[:, :, u]
                fh.write(f"MOTIF motif_{u}\n")
                fh.write(f"letter-probability matrix: alength= {len(alphabet)} w= {motif.shape[0]}\n")
                for k in range(motif.shape[0]):
                    fh.write("  ".join([f"{x:.6f}" for x in motif[k,:]]) + "\n")
                fh.write("\n")
            fh.write("\n")



@dataclass
class MultiTrainingEvaluation():
    """ Class to store results of multiple training evaluations. """
    trainings: list[TrainingEvaluation] = field(default_factory=list)

    def add_result(self, runID, 
                   motifs: MotifWrapper,
                   links: list[Links.MultiLink], 
                   hg_genome: sr.Genome, 
                   time: float = -1):
        """ Add results of a single exon training run to the class. 
        Parameters:
            runID: any - anything that lets us identify the run later
            motifs: np.ndarray - numpy array of motifs used in the training run
            links: list[Links.MultiLink] - list of MultiLinks found in the training run
            hg_genome: sr.Genome - Human genome used in the training run, should contain the human sequence(s) that
                                   were used in the training run
            time: float - time in seconds, -1 if not available
        """
        assert typecheck(hg_genome, 'Genome', die=True)
        assert hg_genome.species == 'Homo_sapiens'
        
        nhumanExons = 0
        nhumanOccs = 0
        nexonsHit = 0
        nlinksThatHit = 0
        humanOccsThatHit = set()
        for link in links:
            assert typecheck(link, "MultiLink", die=True), f"Unexpected type of link (want 'MultiLink'): {type(link)}"
            occs = link.getSpeciesOccurrences('Homo_sapiens')
            if occs is not None:
                nhumanOccs += len(occs)

        for hg_sequence in hg_genome:
            hg_sequence: sr.Sequence = hg_sequence # for linting
            for exon in hg_sequence.genomic_elements:
                nhumanExons += 1
                exonHit = False
                exonStart, exonEnd = exon.getRelativePositions(hg_sequence)
                for link in links:
                    hgoccs = link.getSpeciesOccurrences('Homo_sapiens')
                    if hgoccs is not None:
                        assert len(set([o.sequence.species for o in hgoccs])) == 1, \
                            f"Not all Occurrences from {hgoccs=} have same species: {link}" # should never happen
                        # determine hg-hitting links by creating new MultiLinks with only one hg occ at a time and count
                        remainingOccs = [o for occs in link.occs for o in occs \
                                            if o.sequence.species != 'Homo_sapiens']
                        for hgocc in hgoccs:
                            # only need to consider hg exon hitting occs
                            if (hgocc.sequence == hg_sequence) and (exonStart <= hgocc.position + hgocc.sitelen - 1) \
                                    and (hgocc.position < exonEnd):
                                #nhumanOccsThatHit += 1
                                humanOccsThatHit.add(hgocc)
                                # create new MultiLink with only the one human Occurrence 
                                #   to see how many unique links would hit
                                if len(remainingOccs) == 0:
                                    continue
                                if link.singleProfile():
                                    exonHit = True
                                    mh = Links.MultiLink([hgocc]+remainingOccs, singleProfile=True)
                                    nlinksThatHit += mh.nUniqueLinks()
                                else:
                                    profileoccs = [hgocc] \
                                                    + [o for o in remainingOccs if o.profileIdx == hgocc.profileIdx]
                                    if len(profileoccs) > 1:
                                        exonHit = True
                                        mh = Links.MultiLink(profileoccs, singleProfile=True)
                                        nlinksThatHit += mh.nUniqueLinks()
                
                if exonHit:
                    nexonsHit += 1

        self.trainings.append(
            TrainingEvaluation(runID=runID, 
                               motifs=motifs, 
                               links=links, 
                               nexonsHit=nexonsHit, 
                               nhumanExons=nhumanExons, 
                               nhumanOccs=nhumanOccs,
                               nhumanOccsThatHit=len(humanOccsThatHit),
                               nlinksThatHit=nlinksThatHit, 
                               trainingTime=time)
            )


    def accuracy(self):
        """ Returns a tuple (sensitivity, specificity) """
        nexonsHit = sum([t.nexonsHit for t in self.trainings])
        nhumanExons = sum([t.nhumanExons for t in self.trainings])
        nhumanOccs = sum([t.nhumanOccs for t in self.trainings])
        nhumanOccsThatHit = sum([t.nhumanOccsThatHit for t in self.trainings])
        nlinks = sum([t.nlinks for t in self.trainings])
        nlinksThatHit = sum([t.nlinksThatHit for t in self.trainings])
        sensitivity = nexonsHit/nhumanExons
        specificity = nhumanOccsThatHit/nhumanOccs if nhumanOccs > 0 else None
        link_specificity = nlinksThatHit/nlinks if nlinks > 0 else None
        logging.info("[MultiTrainingEvaluation.accuracy] >>> " + \
                        f"Sensitivity: {nexonsHit} / {nhumanExons} = {sensitivity}")
        logging.info("[MultiTrainingEvaluation.accuracy] >>> " + \
                        f"Specificity: {nhumanOccsThatHit} / {nhumanOccs} = {specificity}") # TODO: why do these numbers differ from link_sp for normal Links???
        logging.info("[MultiTrainingEvaluation.accuracy] >>> " + \
                        f"Link Specificity: {nlinksThatHit} / {nlinks} = {link_specificity}")
        return (sensitivity, specificity)


    def dump(self, filename):
        """ Write a JSON representation of the object to a file. """
        
        d = [t.toDict() for t in self.trainings]
        with open(filename, 'wt') as fh:
            json.dump(d, fh)



def loadMultiTrainingEvaluation(filename, allGenomes: list[sr.Genome], 
                                recalculate: bool = False) -> MultiTrainingEvaluation:
    """ Load a MultiTrainingEvaluation object from a JSON file. `allGenomes` must contain all Sequences (as a list of
    Genome objects) that were used in the training runs. 
    If `recalculate` is True, only the links and training time are loaded per training run and the metrics are 
    re-computed via MultiTrainingEvaluation.add_result(). This is useful if the metrics changed or were added later. """
    with open(filename, "rt") as fh:
        d = json.load(fh)

    all_ids = [s.id for g in allGenomes for s in g]
    if len(all_ids) != len(set(all_ids)):
        idcount = {id: 0 for id in all_ids}
        for id in all_ids:
            idcount[id] += 1
        logging.warning("[loadMultiTrainingEvaluation] >>> Duplicate sequence IDs found in allGenomes: " \
                        + ", ".join([f"{id} ({count}x)" for id, count in idcount.items() if count > 1]))

    seqidToSeq = {seq.id: seq for genome in allGenomes for seq in genome} # needed to avoid repeated nested searches
    for t in d:
        for l in t['links']:
            if l['classname'] == 'Link':
                for occ in l['occs']:
                    assert occ[0] in seqidToSeq, f"[ERROR] >>> Sequence {occ[0]} not found in allGenomes."
            else:
                for occs in l['occs']:
                    for occ in occs:
                        assert occ[0] in seqidToSeq, f"[ERROR] >>> Sequence {occ[0]} not found in allGenomes."

    mte = MultiTrainingEvaluation()
    for t in d:
        links = []
        if len(t['links']) > 0:
            for l in t['links']:
                if l['classname'] == 'Link':
                    s = [seqidToSeq[occ[0]] for occ in l['occs']]
                else:
                    s = [[seqidToSeq[occ[0]] for occ in occs] for occs in l['occs']]

                links.append(Links.linkFromDict_fast(l, s)) # type: ignore

        if recalculate:
            hgGenome = [g for g in allGenomes if g.species == 'Homo_sapiens']
            assert len(hgGenome) == 1, f"[ERROR] >>> found {len(hgGenome)} human genomes: {hgGenome}"
            hgGenome = hgGenome[0]
            mte.add_result(t['runID'],
                           motifs=loadMotifWrapperFromDict(t['motifs']),
                           links=links,
                           hg_genome=hgGenome,
                           time=t['trainingTime'])
        else:
            mte.trainings.append(
                TrainingEvaluation(runID=t['runID'], 
                                   motifs=loadMotifWrapperFromDict(t['motifs']),
                                   links=links,
                                   nexonsHit=t['nexonsHit'],
                                   nhumanExons=t['nhumanExons'],
                                   nhumanOccs=t['nhumanOccs'],
                                   nhumanOccsThatHit=t['nhumanOccsThatHit'],
                                   nlinksThatHit=t['nlinksThatHit'],
                                   trainingTime=t['trainingTime'])
                )
            
    return mte



def trainAndEvaluate(runID,
                     trainsetup: setup.ProfileFindingTrainingSetup, 
                     evaluator: MultiTrainingEvaluation,
                     outdir: str, outprefix: str = "",
                     # trainingWithReporting: bool = True,
                     do_not_train: bool = False,
                     rand_seed: int = None) -> None: # type: ignore
    """ 
    Train profiles on a given training setup and evaluate them. 
    Parameters:
        runID: any
            Identifier for the training run.
        trainsetup: ProfileFindingTrainingSetup
            Training setup to use for training and evaluation.
        evaluator: MultiTrainingEvaluation
            To store the results of the training run.
        outdir: str
            Directory to save resulting plots to. Set to None for no saving.
        outprefix: str
            Prefix for output files.
        # trainingWithReporting: bool
        #     If True, training is done with reporting, otherwise with classic training.
        do_not_train: bool
            If True, the model is not trained, but only the best initial profiles are returned.
        rand_seed: int
            Random seed for reproducibility.

    Returns:
        None
    """

    assert trainsetup.initProfiles is not None, "[ERROR] >>> No seed profiles found in trainsetup."

    # build and randomly initialize profile model
    tf.keras.backend.clear_session()  # type: ignore # avoid memory cluttering by remains of old models
    logging.info(f"[training.trainAndEvaluate] >>> Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    specProModel = model.SpecificProfile(setup = trainsetup,
                                         rand_seed = rand_seed)

    # start training
    start = time()
    try:
        if not do_not_train:
             specProModel.train(verbose_freq=10)
        else:
            # do not train, but return best initial profiles
            for _ in range(trainsetup.n_best_profiles):
                mean_losses = specProModel.get_mean_losses(specProModel.data.getDataset(withPosTracking = True), 
                                                        specProModel.getP(), specProModel.P_logit) # (U)
                best_profile = tf.argmin(mean_losses).numpy()
                specProModel.profile_cleanup(best_profile, 0)

        end = time()
    except Exception as e:
        end = time()
        logging.error(f"[training.trainAndEvaluate] >>> Training failed after {end-start:.2f}.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        logging.debug(full_stack())
        
    training_time = end-start
    logging.info(f"[training.trainAndEvaluate] >>> Training time: {training_time:.2f}")

    # evaluate model (if possible)
    try:
        if outdir is not None:
            # draw training history
            fig, _ = plotting.plotHistory(specProModel.history)
            fig.savefig(os.path.join(outdir, outprefix+"training_history.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, _ = plotting.plotProfileLossHeatmap(specProModel.profile_tracking)
            fig.savefig(os.path.join(outdir, outprefix+"profile_loss_history_heat.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig = plotting.plotBestProfileLossScatter(specProModel.profile_tracking)
            fig.write_image(os.path.join(outdir, outprefix+"profile_loss_history_scat.png"), width=1920, height=1080)

        # get match sites of profiles and create Links

        profile_report = specProModel.profile_report
        motifs = MotifWrapper(profile_report.P, trainsetup.data.alphabet,
                              {'Pthresh': [float(pt) for pt in profile_report.threshold],  # type: ignore
                               'Ploss': [float(pl) for pl in profile_report.loss]})
        sites, sitescores = specProModel.get_profile_match_sites(trainsetup.data.getDataset(withPosTracking = True, 
                                                                                            original_data=True),
                                                                 profile_report.P, profile_report.threshold)
        occurrences = trainsetup.data.convertModelSites(sites.numpy(), trainsetup.k) # type: ignore

        #logging.debug(f"[training.trainAndEvaluate] >>> sites: {sites.numpy()[:20,]}") # type: ignore # (sites, (genomeID, contigID, pos, u, f))
        mlinks = Links.multiLinksFromOccurrences(occurrences)
        #logging.debug(f"[training.trainAndEvaluate] >>> mlinks[:{min(len(mlinks), 2)}] {mlinks[:min(len(mlinks), 2)]}")
        logging.debug(f"[training.trainAndEvaluate] >>> len(mlinks) {len(mlinks)}")
        logging.debug(f"[training.trainAndEvaluate] >>> number of unique links: {Links.nLinks(mlinks)}") # type: ignore

        # add evaluation results to evaluator
        hgGenome = [g for g in trainsetup.data.training_data.getGenomes() if g.species == 'Homo_sapiens']
        assert len(hgGenome) <= 1, f"[ERROR] >>> found {len(hgGenome)} human genomes: {hgGenome}"
        if len(hgGenome) == 1:
            # do normal evaluation
            hgGenome = hgGenome[0]
            evaluator.add_result(runID, motifs, mlinks, hgGenome, training_time)
        else:
            # assume there is no exon data with human reference and everything, bypass evaluation but still store motifs
            logging.warning("[training.trainAndEvaluate] >>> Training data does not contain human genomes, " \
                            + "bypassing evaluation")
            evaluator.trainings.append(
                TrainingEvaluation(runID, motifs, mlinks, 0, 0, 0, 0, 0, training_time)
            )

        # draw link image
        kmerSites: list[Links.Occurrence] = []
        for kmer in trainsetup.initKmerPositions:
            kmerSites.extend(trainsetup.initKmerPositions[kmer])
        maskSites = trainsetup.data.convertModelSites(specProModel.profile_report.masked_sites,
                                                      sitelen = trainsetup.k)
        img = plotting.drawGeneLinks(mlinks,  # type: ignore
                                     trainsetup.data.training_data.getGenomes(), # not really needed, but defines genome order
                                     imname=os.path.join(outdir, outprefix+"links.png") if outdir is not None else None, 
                                     kmerSites=kmerSites, kmerCol='deeppink',
                                     maskingSites=maskSites, maskingCol='chocolate',
                                     show=False)
        img.close()
        
    except Exception as e:
        logging.error("[training.trainAndEvaluate] >>> Evaluation failed.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        logging.debug(full_stack())



def _linkImg(links: list[Links.MultiLink], genomes: list[Links.sr.Genome], imname: str,
             splitthreshold: int = 150, splitsize: int = 100,
             single_genecol: str = None, single_linkcol: str = None,
             **kwargs):
    """ Call plotting.drawGeneLinks() with the given MultiLinks and save the image. Splits into multiple plots if too
     many genomes 
     
     Parameters:
        links: list[Links.MultiLink]
            List of MultiLinks to draw.
        genomes: list[Links.sr.Genome]
            List of Genomes to draw the links on.
        imname: str
            Name of the image file to save.
        splitthreshold: int
            Number of genomes at which to split the image into multiple parts.
        splitsize: int
            Number of genomes per part.
        single_genecol: str
            Optional single color to use for all genes in the link plots.
        single_linkcol: str
            Optional single color to use for all links in the link plots.
        kwargs: dict
            Additional keyword arguments for plotting.drawGeneLinks().
    """
    if len(genomes) > splitthreshold:
        # split images into multiple parts of 500 genes each
        n_parts = len(genomes) / splitsize
        n_parts = int(n_parts) if n_parts == int(n_parts) else int(n_parts) + 1
        genomeparts = []
        for i in range(n_parts):
            genomeparts.append(genomes[i*splitsize:min(len(genomes), (i+1)*splitsize)])
    else:
        genomeparts = [genomes]

    suffix = Path(imname).suffix
    imname_pre = imname[:-len(suffix)]
    for i, genomes in enumerate(genomeparts):
        part = f"_part{i}" if len(genomeparts) > 1 else ""
        genecols = [single_genecol]*len(genomes) if single_genecol is not None else None
        linkcols = [single_linkcol]*len(links) if single_linkcol is not None else None
        img = plotting.drawGeneLinks(links,  # type: ignore
                                     genomes, # not really needed, but defines genome order
                                     imname=imname_pre + part + suffix,
                                     genecols=genecols,
                                     linkcols=linkcols,
                                     **kwargs)
        img.close()


def trainAndTest(runID,
                 trainsetup: setup.ProfileFindingTrainingSetup, 
                 testdata: ModelDataSet,
                 train_evaluator: MultiTrainingEvaluation,
                 test_evaluator: MultiTrainingEvaluation,
                 outdir: str, outprefix: str = "",
                 do_not_train: bool = False,
                 rand_seed: int = None,
                 linkplot_splitthreshold: int = 150,
                 linkplot_splitsize: int = 100,
                 linkplot_single_genecol: str = None,
                 linkplot_single_linkcol: str = None,
                 linkplot_genewidth: int = 20,
                 linkplot_linkwidth: int = 10,
                 **linkplot_kwargs) -> None: # type: ignore
    """ 
    Train profiles on a given training setup and evaluate them. 
    Parameters:
        runID: any
            Identifier for the training run.
        trainsetup: ProfileFindingTrainingSetup
            Training setup to use for training and evaluation.
        testdata: ModelDataSet
            Data to test the trained profiles on.
        train_evaluator: MultiTrainingEvaluation
            To store the results of the training run.
        test_evaluator: MultiTrainingEvaluation
            To store the results of the trained profiles on the test data.
        outdir: str
            Directory to save resulting plots to. Set to None for no saving.
        outprefix: str
            Prefix for output files.
        # trainingWithReporting: bool
        #     If True, training is done with reporting, otherwise with classic training.
        do_not_train: bool
            If True, the model is not trained, but only the best initial profiles are returned.
        rand_seed: int
            Random seed for reproducibility.
        linkplot_splitthreshold: int
            Number of genomes at which to split the link image into multiple parts.
        linkplot_splitsize: int
            Number of genomes per link image part.
        linkplot_single_genecol: str
            Optional single color to use for all genes in the link plots.
        linkplot_single_linkcol: str
            Optional single color to use for all links in the link plots.
        linkplot_genewidth: int
            Width of gene lines in the link plots.
        linkplot_linkwidth: int
            Width of link lines in the link plots.
        linkplot_kwargs: dict
            Additional keyword arguments for plotting.drawGeneLinks().

    Returns:
        None
    """

    assert trainsetup.initProfiles is not None, "[ERROR] >>> No seed profiles found in trainsetup."

    # build and randomly initialize profile model
    tf.keras.backend.clear_session()  # type: ignore # avoid memory cluttering by remains of old models
    logging.info(f"[training.trainAndTest] >>> Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    specProModel = model.SpecificProfile(setup = trainsetup,
                                         rand_seed = rand_seed)

    # start training
    start = time()
    try:
        if not do_not_train:
             specProModel.train(verbose_freq=10)
        else:
            # do not train, but return best initial profiles
            for _ in range(trainsetup.n_best_profiles):
                mean_losses = specProModel.get_mean_losses(specProModel.data.getDataset(withPosTracking = True), 
                                                           specProModel.getP(), specProModel.P_logit) # (U)
                best_profile = tf.argmin(mean_losses).numpy()
                specProModel.profile_cleanup(best_profile, 0)

        end = time()
    except Exception as e:
        end = time()
        logging.error(f"[training.trainAndTest] >>> Training failed after {end-start:.2f}.")
        logging.error(f"[training.trainAndTest] >>> Exception:\n{e}")
        logging.debug(full_stack())
        
    training_time = end-start
    logging.info(f"[training.trainAndTest] >>> Training time: {training_time:.2f}")

    # plot history if possible
    if outdir is not None:
        try:
            logging.info(f"[training.trainAndTest] >>> Plotting history")
            # draw training history
            fig, _ = plotting.plotHistory(specProModel.history)
            fig.savefig(os.path.join(outdir, outprefix+"training_history.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, _ = plotting.plotProfileLossHeatmap(specProModel.profile_tracking)
            fig.savefig(os.path.join(outdir, outprefix+"profile_loss_history_heat.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig = plotting.plotBestProfileLossScatter(specProModel.profile_tracking)
            fig.write_image(os.path.join(outdir, outprefix+"profile_loss_history_scat.png"), width=1920, height=1080)
        except Exception as e:
            logging.error("[training.trainAndTest] >>> Plotting failed.")
            logging.error(f"[training.trainAndTest] >>> Exception:\n{e}")
            logging.debug(full_stack())

    # store and plot training data motif sites
    try:
        logging.info(f"[training.trainAndTest] >>> Extracting and storing training info")
        profile_report = specProModel.profile_report
        motifs = MotifWrapper(profile_report.P, trainsetup.data.alphabet,
                              {'Pthresh': [float(pt) for pt in profile_report.threshold],  # type: ignore
                               'Ploss': [float(pl) for pl in profile_report.loss]})
        train_sites, train_sitescores = specProModel.get_profile_match_sites(
            trainsetup.data.getDataset(withPosTracking = True, 
                                       original_data=True),
            profile_report.P, profile_report.threshold)
        train_occurrences = trainsetup.data.convertModelSites(train_sites.numpy(), trainsetup.k) # type: ignore

        #logging.debug(f"[training.trainAndTest] >>> <training> sites: {train_sites.numpy()[:20,]}") # type: ignore # (sites, (genomeID, contigID, pos, u, f))
        train_mlinks = Links.multiLinksFromOccurrences(train_occurrences)
        #logging.debug(f"[training.trainAndTest] >>> <training> mlinks[:{min(len(train_mlinks), 2)}] {train_mlinks[:min(len(train_mlinks), 2)]}")
        logging.debug(f"[training.trainAndTest] >>> <training> len(mlinks) {len(train_mlinks)}")
        logging.debug(f"[training.trainAndTest] >>> <training> number of unique links: {Links.nLinks(train_mlinks)}") # type: ignore

        train_evaluator.trainings.append(
                TrainingEvaluation(runID, motifs, train_mlinks, 0, 0, 0, 0, 0, training_time)
            )

        if outdir is not None:
            # plot logos
            try:
                logging.info(f"[training.trainAndTest] >>> Plotting logos")
                # create plt axes, one for each motif
                nmotifs = profile_report.P.shape[2]
                fig, axs = plt.subplots(nmotifs, 1, figsize=(16, 9*nmotifs))
                plotting.plotLogo(profile_report.P, trainsetup.data.alphabet, pLosses = profile_report.loss, ax=axs)
                fig.savefig(os.path.join(outdir, outprefix+"logos.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logging.error("[training.trainAndTest] >>> Plotting logos failed.")
                logging.error(f"[training.trainAndTest] >>> Exception:\n{e}")
                logging.debug(full_stack())

            # draw link image
            logging.info(f"[training.trainAndTest] >>> Plotting link image")
            imdir = os.path.join(outdir, "link_plots")
            os.makedirs(imdir, exist_ok=True)
            kmerSites: list[Links.Occurrence] = []
            for kmer in trainsetup.initKmerPositions:
                kmerSites.extend(trainsetup.initKmerPositions[kmer])
            maskSites = trainsetup.data.convertModelSites(specProModel.profile_report.masked_sites,
                                                          sitelen = trainsetup.k)
            # img = plotting.drawGeneLinks(train_mlinks,  # type: ignore
            #                              trainsetup.data.training_data.getGenomes(), # not really needed, but defines genome order
            #                              imname=os.path.join(outdir, outprefix+"training_links.png"),
            #                              kmerSites=kmerSites, maskingSites=maskSites,
            #                              connectLinks=False,
            #                              show=False, 
            #                              genecols=genecols,
            #                              linkcols=linkcols,
            #                              genewidth=linkplot_genewidth,
            #                              linkwidth=linkplot_linkwidth,
            #                              **linkplot_kwargs)
            # img.close()
            _linkImg(train_mlinks, trainsetup.data.training_data.getGenomes(),
                     os.path.join(imdir, outprefix+"training_links.png"),
                     splitthreshold=linkplot_splitthreshold, splitsize=linkplot_splitsize,
                     kmerSites=kmerSites, maskingSites=maskSites, connectLinks=False, show=False,
                     single_genecol=linkplot_single_genecol,
                     single_linkcol=linkplot_single_linkcol,
                     genewidth=linkplot_genewidth,
                     linkwidth=linkplot_linkwidth,
                     **linkplot_kwargs)
            
    except Exception as e:
        logging.error("[training.trainAndTest] >>> Storing training data failed.")
        logging.error(f"[training.trainAndTest] >>> Exception:\n{e}")
        logging.debug(full_stack())

    # run model on test data
    try:
        logging.info(f"[training.trainAndTest] >>> Running model on test data")
        profile_report = specProModel.profile_report
        motifs = MotifWrapper(profile_report.P, trainsetup.data.alphabet,
                              {'Pthresh': [float(pt) for pt in profile_report.threshold],  # type: ignore
                               'Ploss': [float(pl) for pl in profile_report.loss]})
        test_sites, test_sitescores = specProModel.get_profile_match_sites(testdata.getDataset(withPosTracking = True, 
                                                                                               original_data=True),
                                                                           profile_report.P, profile_report.threshold)
        test_occurrences = testdata.convertModelSites(test_sites.numpy(), trainsetup.k) # type: ignore
        #logging.debug(f"[training.trainAndTest] >>> <test> sites: {test_sites.numpy()[:20,]}") # type: ignore # (sites, (genomeID, contigID, pos, u, f))
        test_mlinks = Links.multiLinksFromOccurrences(test_occurrences)
        #logging.debug(f"[training.trainAndTest] >>> <test> mlinks[:{min(len(test_mlinks), 2)}] {test_mlinks[:min(len(test_mlinks), 2)]}")
        logging.debug(f"[training.trainAndTest] >>> <test> len(mlinks) {len(test_mlinks)}")
        logging.debug(f"[training.trainAndTest] >>> <test> number of unique links: {Links.nLinks(test_mlinks)}") # type: ignore

        test_evaluator.trainings.append(
            TrainingEvaluation(runID, motifs, test_mlinks, 0, 0, 0, 0, 0, training_time)
        )

        if outdir is not None:
            # draw link image
            logging.info(f"[training.trainAndTest] >>> Plotting test link image")
            imdir = os.path.join(outdir, "link_plots")
            os.makedirs(imdir, exist_ok=True)
            # img = plotting.drawGeneLinks(test_mlinks,  # type: ignore
            #                              testdata.training_data.getGenomes(), # not really needed, but defines genome order
            #                              imname=os.path.join(outdir, outprefix+"test_links.png"),# \
            #                                 #if outdir is not None else None, 
            #                              connectLinks=False,
            #                              show=False, 
            #                              genecols=genecols,
            #                              linkcols=linkcols,
            #                              genewidth=linkplot_genewidth,
            #                              linkwidth=linkplot_linkwidth,
            #                              **linkplot_kwargs)
            # img.close()
            _linkImg(test_mlinks, testdata.training_data.getGenomes(),
                     os.path.join(imdir, outprefix+"test_links.png"),
                     splitthreshold=linkplot_splitthreshold, splitsize=linkplot_splitsize,
                     connectLinks=False, show=False, 
                     single_genecol=linkplot_single_genecol,
                     single_linkcol=linkplot_single_linkcol,
                     genewidth=linkplot_genewidth,
                     linkwidth=linkplot_linkwidth,
                     **linkplot_kwargs)
    except Exception as e:
        logging.error("[training.trainAndTest] >>> Evaluation on test data failed.")
        logging.error(f"[training.trainAndTest] >>> Exception:\n{e}")
        logging.debug(full_stack())



def testMotifs(runID, motifwrapper: MotifWrapper,
               traindata: ModelDataSet.ModelDataSet,
               testdata: ModelDataSet.ModelDataSet,
               match_score_factor,
               train_evaluator: MultiTrainingEvaluation,
               test_evaluator: MultiTrainingEvaluation,
               outdir: str, outprefix: str = "",
               linkplot_splitthreshold: int = 150,
               linkplot_splitsize: int = 100,
               linkplot_single_genecol: str = None,
               linkplot_single_linkcol: str = None,
               linkplot_genewidth: int = 20,
               linkplot_linkwidth: int = 10,
               **linkplot_kwargs) -> None:
    """ Test the performance of any motifs on the training and test data.
    Parameters:
        runID: any
            Identifier for the training run.
        motifs: np.ndarray with shape (k, alphabet_size, U)
            Motifs to test.
        traindata: ModelDataSet.ModelDataSet
            Training data to test the motifs on.
        testdata: ModelDataSet.ModelDataSet
            Test data to test the motifs on.
        match_score_factor: float
            Factor to multiply the maximum training data score of the motifs with to get the threshold.
        test_evaluator: MultiTrainingEvaluation
            To store the results of the test data evaluation.
        outdir: str
            Directory to save resulting plots to. Set to None for no saving.
        outprefix: str
            Prefix for output files.
        linkplot_splitthreshold: int
            Number of genomes at which to split the link image into multiple parts.
        linkplot_splitsize: int
            Number of genomes per link image part.
        linkplot_single_genecol: str
            Optional single color to use for all genes in the link plots.
        linkplot_single_linkcol: str
            Optional single color to use for all links in the link plots.
        linkplot_genewidth: int
            Width of gene lines in the link plots.
        linkplot_linkwidth: int
            Width of link lines in the link plots.
        linkplot_kwargs: dict
            Additional keyword arguments for plotting.drawGeneLinks().
      """
    assert motifwrapper is not None, "[ERROR] >>> No seed profiles found in trainsetup."
    motifs = motifwrapper.motifs
    assert len(motifs.shape) == 3, f"[ERROR] >>> Expected 3D array, got {motifs.shape}."
    try:
        U = motifs.shape[2]
        k = motifs.shape[0]
        dummy_setup = setup.ProfileFindingTrainingSetup(
            traindata,
            U = U, k = k, 
            midK = k, s = 0, epochs=1, gamma=1.0, l2=0.0, match_score_factor=0, 
            learning_rate=0, lr_patience=0, lr_factor=0, rho=0, sigma=0, profile_plateau=0, profile_plateau_dev=0, 
            n_best_profiles=1, phylo_t=0)
        dummy_setup.initProfiles = motifs # should be the logits, but we don't actually use them so it's fine
        dummy_model = model.SpecificProfile(dummy_setup)
        
        # get max scores of all motifs at all sites in the training data, scores has shape (n_sites, 6)
        score_thresholds = []
        for pIdx in range(motifs.shape[2]):
            _, scores = dummy_model.get_profile_match_sites(traindata.getDataset(withPosTracking=True, original_data=True), 
                                                            motifs, score_threshold=0.0, pIdx=pIdx)
            score_thresholds.append(np.max(scores) * match_score_factor)

        score_thresholds = np.array(score_thresholds)
        # evaluate on training data
        train_sites, train_scores = dummy_model.get_profile_match_sites(traindata.getDataset(withPosTracking = True, 
                                                                                             original_data=True), 
                                                                        motifs, score_threshold=score_thresholds)
        train_occurrences = traindata.convertModelSites(train_sites.numpy(), k)
        train_mlinks = Links.multiLinksFromOccurrences(train_occurrences)
        train_evaluator.trainings.append(
            TrainingEvaluation(runID, motifwrapper, train_mlinks, 0, 0, 0, 0, 0, 0)
        )

        # evaluate on test data
        test_sites, test_scores = dummy_model.get_profile_match_sites(testdata.getDataset(withPosTracking = True, 
                                                                                          original_data=True), 
                                                                      motifs, score_threshold=score_thresholds)
        test_occurrences = testdata.convertModelSites(test_sites.numpy(), k)
        test_mlinks = Links.multiLinksFromOccurrences(test_occurrences)
        test_evaluator.trainings.append(
            TrainingEvaluation(runID, motifwrapper, test_mlinks, 0, 0, 0, 0, 0, 0)
        )

        if outdir is not None:
            # draw link images
            logging.info(f"[training.testMotifs] >>> Plotting train link image")
            imdir = os.path.join(outdir, "link_plots")
            os.makedirs(imdir, exist_ok=True)
            train_genomes = traindata.training_data.getGenomes()
            # if len(train_genomes > 700):
            #     # split images into multiple parts of 500 genes each
            #     n_parts = (len(train_genomes) // 500) + 1
            #     genomeparts = []
            #     for i in range(n_parts):
            #         genomeparts.append(train_genomes[i*500:min(len(train_genomes), (i+1)*500)])
            # else:
            #     genomeparts = [train_genomes]

            # for i, genomes in enumerate(genomeparts):
            #     part = f"_part{i}" if len(genomeparts) > 1 else ""
            #     genecols = [linkplot_single_genecol]*len(genomes) if linkplot_single_genecol is not None else None
            #     linkcols = [linkplot_single_linkcol]*len(train_mlinks) if linkplot_single_linkcol is not None else None
            #     img = plotting.drawGeneLinks(train_mlinks,  # type: ignore
            #                                  genomes, # not really needed, but defines genome order
            #                                  imname=os.path.join(outdir, outprefix+f"train_links{part}.png"),
            #                                  connectLinks=False,
            #                                  show=False, 
            #                                  genecols=genecols,
            #                                  linkcols=linkcols,
            #                                  genewidth=linkplot_genewidth,
            #                                  linkwidth=linkplot_linkwidth,
            #                                  **linkplot_kwargs)
            #     img.close()
            _linkImg(train_mlinks, train_genomes,
                     os.path.join(imdir, outprefix+"train_links.png"),
                     connectLinks=False, show=False, 
                     splitthreshold=linkplot_splitthreshold, splitsize=linkplot_splitsize,
                     single_genecol=linkplot_single_genecol,
                     single_linkcol=linkplot_single_linkcol,
                     genewidth=linkplot_genewidth,
                     linkwidth=linkplot_linkwidth,
                     **linkplot_kwargs)

            logging.info(f"[training.testMotifs] >>> Plotting test link image")
            test_genomes = testdata.training_data.getGenomes()
            # if len(test_genomes > 700):
            #     # split images into multiple parts of 500 genes each
            #     n_parts = (len(test_genomes) // 500) + 1
            #     genomeparts = []
            #     for i in range(n_parts):
            #         genomeparts.append(test_genomes[i*500:min(len(test_genomes), (i+1)*500)])
            # else:
            #     genomeparts = [test_genomes]

            # for i, genomes in enumerate(genomeparts):
            #     part = f"_part{i}" if len(genomeparts) > 1 else ""
            #     genecols = [linkplot_single_genecol]*len(genomes) if linkplot_single_genecol is not None else None
            #     linkcols = [linkplot_single_linkcol]*len(test_mlinks) if linkplot_single_linkcol is not None else None
            #     img = plotting.drawGeneLinks(test_mlinks,  # type: ignore
            #                                  genomes, # not really needed, but defines genome order
            #                                  imname=os.path.join(outdir, outprefix+f"test_links{part}.png"),
            #                                  connectLinks=False,
            #                                  show=False,
            #                                  genecols=genecols,
            #                                  linkcols=linkcols,
            #                                  genewidth=linkplot_genewidth,
            #                                  linkwidth=linkplot_linkwidth,
            #                                  **linkplot_kwargs)
            #     img.close()
            _linkImg(test_mlinks, test_genomes,
                     os.path.join(imdir, outprefix+"test_links.png"),
                     connectLinks=False, show=False, 
                     splitthreshold=linkplot_splitthreshold, splitsize=linkplot_splitsize,
                     single_genecol=linkplot_single_genecol,
                     single_linkcol=linkplot_single_linkcol,
                     genewidth=linkplot_genewidth,
                     linkwidth=linkplot_linkwidth,
                     **linkplot_kwargs)

    except Exception as e:
        logging.error("[training.testMotifs] >>> Evaluation on test data failed.")
        logging.error(f"[training.testMotifs] >>> Exception:\n{e}")
        logging.debug(full_stack())