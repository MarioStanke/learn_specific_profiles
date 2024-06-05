from dataclasses import dataclass, field
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px
import tensorflow as tf
from time import time

from . import Links
from . import model_new as model
from . import plotting
from . import ProfileFindingSetup as setup
from . import SequenceRepresentation as sr
from .typecheck import typecheck, typecheck_list


@dataclass
class MotifWrapper():
    """ Bascially store the motifs np.ndarray together with any metadata. TODO: do this properly """
    motifs: np.ndarray
    metadata: str = None # can be anything for now, just make sure it is JSON-serializable

    def toDict(self) -> dict:
        # catch tensors, try to convert them to numpy first
        if callable(getattr(self.motifs, 'numpy', None)):
            try:
                motifs = self.motifs.numpy()
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

        try:
            _ = json.dumps(self.metadata)
            return {
                'motifs': motifs,
                'metadata': self.metadata
            }
        except Exception as e:
            logging.error(f"[MotifWrapper.toDict] Error converting metadata to JSON.")
            logging.error(f"[MotifWrapper.toDict] Exception:\n{e}")
            logging.info("[MotifWrapper.toDict] Not returning metadata")
            return {
                'motifs': motifs
            }


def loadMotifWrapperFromDict(d: dict) -> MotifWrapper:
    return MotifWrapper(np.asarray(d['motifs']), d['metadata'])



@dataclass 
class TrainingEvaluation():
    """ Class to store results of a single training evaluation. """
    runID: str # anything that lets us identify the run later
    motifs: MotifWrapper # motifs found in the training run
    links: list[Links.Link]
    nexonsHit: int
    nhumanExons: int
    nhumanOccs: int # number of occurrences in human
    nhumanOccsThatHit: int # number of occurrences in human that hit an exon
    nlinks: int # in case of MultiLinks, this could be the number of nUniqueLinks, thus different from len(links)
    nlinksThatHit: int
    trainingTime: float # time in seconds, TODO: put that in a history class or sth, does not really fit here

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



@dataclass
class MultiTrainingEvaluation():
    """ Class to store results of multiple training evaluations. """
    trainings: list[TrainingEvaluation] = field(default_factory=list)

    def add_result(self, runID, 
                   motifs: MotifWrapper,
                   links: list[Links.Link | Links.MultiLink], 
                   hg_genome: sr.Genome, 
                   time: float = -1):
        """ Add results of a single exon training run to the class. 
        Parameters:
            runID: any - anything that lets us identify the run later
            motifs: np.ndarray - numpy array of motifs used in the training run
            links: list[Links.Link | Links.MultiLink] - list of links found in the training run
            hg_genome: sr.Genome - Human genome used in the training run, should contain the human sequence(s) that
                                   were used in the training run
            time: float - time in seconds, -1 if not available
        """
        assert typecheck(hg_genome, 'Genome', die=True)
        assert hg_genome.species == 'Homo_sapiens'
        
        nlinks = 0
        nhumanExons = 0
        nhumanOccs = 0
        nexonsHit = 0
        nlinksThatHit = 0
        humanOccsThatHit = set()
        #nhumanOccsThatHit = 0
        for link in links:
            assert typecheck_list(link, ["Link", "MultiLink"], die=True), \
                    f"Unexpected type of link (want either 'Link' or 'MultiLink'): {type(link)}"
            nlinks += 1 if link.classname == "Link" else int(link.nUniqueLinks())
            if link.classname == "Link":
                if link.getSpeciesOccurrence('Homo_sapiens') is not None:
                    nhumanOccs += 1
            else:
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
                    if link.classname == "Link":
                        occ = link.getSpeciesOccurrence('Homo_sapiens')
                        if (occ is not None) and (occ.sequence == hg_sequence) \
                            and (exonStart <= occ.position + link.span - 1) and (occ.position < exonEnd):
                            exonHit = True
                            nlinksThatHit += 1
                            #nhumanOccsThatHit += 1
                            humanOccsThatHit.add(occ)
                    else:
                        link: Links.MultiLink = link # for linting
                        hgoccs = link.getSpeciesOccurrences('Homo_sapiens')
                        if hgoccs is not None:
                            assert len(set([o.sequence.species for o in hgoccs])) == 1, \
                                f"Not all Occurrences from {hgoccs=} have same species: {link}" # should never happen
                            
                            remainingOccs = [o for occs in link.occs for o in occs \
                                             if o.sequence.species != 'Homo_sapiens']
                            for hgocc in hgoccs:
                                if (hgocc.sequence == hg_sequence) and (exonStart <= hgocc.position + link.span - 1) \
                                        and (hgocc.position < exonEnd):
                                    #nhumanOccsThatHit += 1
                                    humanOccsThatHit.add(hgocc)
                                    # create new MultiLink with only the one human Occurrence 
                                    #   to see how many unique links would hit
                                    if link.singleProfile():
                                        exonHit = True
                                        mh = Links.MultiLink([hgocc]+remainingOccs, link.span, singleProfile=True)
                                        nlinksThatHit += mh.nUniqueLinks()
                                    else:
                                        profileoccs = [hgocc] \
                                                        + [o for o in remainingOccs if o.profileIdx == hgocc.profileIdx]
                                        if len(profileoccs) > 1:
                                            exonHit = True
                                            mh = Links.MultiLink(profileoccs, link.span, singleProfile=True)
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
                               #nhumanOccsThatHit=nhumanOccsThatHit,
                               nhumanOccsThatHit=len(humanOccsThatHit),
                               nlinks=nlinks, 
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

    # seqIDsToGenomes = {allGenomes[gidx][sidx].id: (gidx, sidx) \
    #                        for gidx in range(len(allGenomes)) \
    #                        for sidx in range(len(allGenomes[gidx]))}

    mte = MultiTrainingEvaluation()
    for t in d:
        if len(t['links']) == 0:
            links = []
        else:
            links = [Links.linkFromDict(l, allGenomes) for l in t['links']]

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
                                   nlinks=t['nlinks'],
                                   nlinksThatHit=t['nlinksThatHit'],
                                   trainingTime=t['trainingTime'])
                )

    return mte



def trainAndEvaluate(runID,
                     trainsetup: setup.ProfileFindingTrainingSetup, 
                     evaluator: MultiTrainingEvaluation,
                     outdir: str, outprefix: str = "",
                     trainingWithReporting: bool = True,
                     rand_seed: int = None) -> None:
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
            Directory to save results to.
        outprefix: str
            Prefix for output files.
        trainingWithReporting: bool
            If True, training is done with reporting, otherwise with classic training.
        rand_seed: int
            Random seed for reproducibility.

    Returns:
        None
        [DEPRECATED] tuple[float, float]: (sensitivity, specificity) or None, None if something went wrong.
    """

    assert trainsetup.initProfiles is not None, "[ERROR] >>> No seed profiles found in trainsetup."

    # build and randomly initialize profile model
    tf.keras.backend.clear_session() # avoid memory cluttering by remains of old models
    specProModel = model.SpecificProfile(setup = trainsetup,
                                         rand_seed = rand_seed)

    # start training
    start = time()
    try:
        specProModel.train(verbose_freq=10)
        end = time()
    except Exception as e:
        end = time()
        logging.error(f"[training.trainAndEvaluate] >>> Training failed after {end-start:.2f}.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        
    training_time = end-start
    logging.info(f"[training.trainAndEvaluate] >>> Training time: {training_time:.2f}")

    assert False # try until here, plotting and evaluation still have to be adapted to the new model

    # evaluate model (if possible)
    try:
        fig, _ = plotting.plotHistory(specProModel.history)
        fig.savefig(os.path.join(outdir, outprefix+"training_history.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        if trainingWithReporting:
            plh = tf.transpose(specProModel.P_report_plosshist).numpy()
            fig, ax = plt.subplots(1,1, figsize=(2*1920/100,2*1080/100), dpi=100)
            hm = ax.imshow(plh, interpolation = 'nearest', cmap="rainbow")
            plt.colorbar(hm)
            fig.savefig(os.path.join(outdir, outprefix+"profile_loss_history_heat.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            bestlosshist = [l.numpy() for l in specProModel.P_report_bestlosshist]
            bestlosshistIdx = [i.numpy() for i in specProModel.P_report_bestlosshistIdx]
            
            px.scatter(x = list(range(len(bestlosshist))),
                       y = bestlosshist, 
                       color = bestlosshistIdx).write_image(os.path.join(outdir, 
                                                                         outprefix+"profile_loss_history_scat.png"), 
                                                            width=1920, height=1080)
            
        # draw link image and calculate accuracy

        # get match sites of profiles
        if trainingWithReporting:
            P, Pthresh, Ploss = specProModel.getP_report()
            motifs = MotifWrapper(P, {'Pthresh': [float(pt) for pt in Pthresh], 
                                      'Ploss': [float(pl) for pl in Ploss]})
        else:
            P, scores, losses = specProModel.getP_optimal(0, lossStatistics=True)
            motifs = MotifWrapper(P, {'scores': scores.tolist(), 'losses': losses.tolist()}) # untested, scores and losses _should_ be numpy arrays but who knows...

        thresh = Pthresh if trainingWithReporting else specProModel.setup.match_score_factor * scores

        onlyPid = None
        if onlyPid is None:
            sites, _, _ = specProModel.get_profile_match_sites(
                specProModel.setup.getDataset(withPosTracking = True, original_data=True),
                thresh, otherP = P)
        else:
            sites, _, _ = specProModel.get_profile_match_sites(
                specProModel.setup.getDataset(withPosTracking = True, original_data=True), 
                thresh[onlyPid], otherP = P[:,:,onlyPid:onlyPid+1])
            
        logging.debug(f"[training.trainAndEvaluate] >>> sites: {sites.numpy()[:20,]}") # (sites, (genomeID, contigID, pos, u, f))
        lfsr = Links.linksFromSites(sites, specProModel.setup.k*3, specProModel.setup.data.genomes, 1000)
        links = lfsr.links
        #linkProfiles = lfsr.linkProfiles
        #skipped = lfsr.skipped
        logging.debug(f"[training.trainAndEvaluate] >>> links[:{min(len(links), 2)}] {links[:min(len(links), 2)]}")
        logging.debug(f"[training.trainAndEvaluate] >>> len(links) {len(links)}")

        hgGenome = [g for g in trainsetup.data.genomes if g.species == 'Homo_sapiens']
        assert len(hgGenome) == 1, f"[ERROR] >>> found {len(hgGenome)} human genomes: {hgGenome}"
        hgGenome = hgGenome[0]
        evaluator.add_result(runID, motifs, links, hgGenome, training_time)

        kmerSites = []
        for kmer in trainsetup.initKmerPositions:
            kmerSites.extend(trainsetup.initKmerPositions[kmer])

        masksides = specProModel.getMaskedSites(0)

        img = plotting.drawGeneLinks(trainsetup.data.genomes, links, 
                                     imname=os.path.join(outdir, outprefix+"links.png"), 
                                     kmerSites=kmerSites, kmerCol='deeppink',
                                     maskingSites=masksides, maskingCol='chocolate',
                                     show=False)
        img.close()
        
    except Exception as e:
        logging.error("[training.trainAndEvaluate] >>> Evaluation failed.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")