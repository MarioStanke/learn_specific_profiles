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
from . import model
from . import plotting
from . import ProfileFindingSetup as setup
from . import SequenceRepresentation as sr
from . import sequtils as su


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
    # runIDs: list = field(default_factory=list) # anything that lets us identify the run later
    # motifs: list[MotifWrapper] = field(default_factory=list) # list of motifs used in the training run
    # links: list[list[Links.Link]] = field(default_factory=list)
    # nexonsHit: list[int] = field(default_factory=list)
    # nhumanExons: list[int] = field(default_factory=list)
    # nlinks: list[int] = field(default_factory=list)
    # nlinksThatHit: list[int] = field(default_factory=list)
    # trainingTime: list[float] = field(default_factory=list) # time in seconds, TODO: put that in a history class or sth, does not really fit here
    trainings: list[TrainingEvaluation] = field(default_factory=list)

    def add_result(self, runID, 
                   motifs: MotifWrapper,
                   links: list[Links.Link | Links.MultiLink], 
                   genomes: list[sr.Genome], 
                   time: float = -1):
        """ Add results of a single exon training run to the class. 
        Parameters:
            runID: any - anything that lets us identify the run later
            motifs: np.ndarray - numpy array of motifs used in the training run
            links: list[Links.Link | Links.MultiLink] - list of links found in the training run
            genomes: list[sr.Genome] - list of genomes used in the training run, must correspond to link occurrences!
            time: float - time in seconds, -1 if not available
        """
        assert genomes[0].species == 'Homo_sapiens'
        
        nlinks = 0
        nhumanExons = 0
        nhumanOccs = 0
        nexonsHit = 0
        nlinksThatHit = 0
        humanOccsThatHit = set()
        #nhumanOccsThatHit = 0
        for link in links:
            assert hasattr(link, "classname") and link.classname in ["Link", "MultiLink"], \
                    f"Unexpected type of link (want either 'Link' or 'MultiLink'): {type(link)}"
            nlinks += 1 if link.classname == "Link" else int(link.nUniqueLinks())
            # TODO: it _should_ hold that occs are sorted and Homo_sapiens is index 0, thus if a link hits human,
            #         occ[0] is from human. It would be cleaner to refactor the Occurrence class fundamentally to just
            #         hold a reference to the Sequence it belongs to.
            if link.classname == "Link":
                if link.genomes[link.occs[0].genomeIdx].species == 'Homo_sapiens':
                    nhumanOccs += 1
            else:
                if link.genomes[link.occs[0][0].genomeIdx].species == 'Homo_sapiens':
                    nhumanOccs += len(link.occs[0])

            #occ = link.occs[0] if link.classname == "Link" else link.occs[0][0]
            #hgoccs = [link.occs[0]] if link.classname == "Link" else link.occs[0]
            #for occ in hgoccs:
            #    if link.genomes[0].species == 'Homo_sapiens':
            #        nlinks += 1
    
        for sequence in genomes[0]:
            for i, exon in enumerate(sequence.genomic_elements):
                nhumanExons += 1
                exonHit = False
                exonStart, exonEnd = exon.getRelativePositions(sequence)
                for link in links:
                    if link.classname == "Link":
                        occ = link.occs[0]
                        # TODO: re-evaluate. I think we cannot guarantee that a link hits human, but count anyway since that would be bad and accuracy should be low then
                        #assert link.genomes[occ.genomeIdx].species == 'Homo_sapiens', \
                        #    f"g: {occ.genomeIdx}, link: {link}"
                        if link.genomes[occ.genomeIdx].species != 'Homo_sapiens':
                            continue

                        if (occ.sequenceIdx == i) and (exonStart <= occ.position + link.span - 1) \
                                and (occ.position < exonEnd):
                            exonHit = True
                            nlinksThatHit += 1
                            #nhumanOccsThatHit += 1
                            humanOccsThatHit.add(occ)
                    else:
                        hgoccs = link.occs[0]
                        assert len(set([o.genomeIdx for o in hgoccs])) == 1, \
                            f"Not all Occurrences from idx 0 have same species: {link}"
                        # TODO: see above
                        if link.genomes[hgoccs[0].genomeIdx].species != 'Homo_sapiens':
                            continue

                        remainingOccs = [o for occs in link.occs[1:] for o in occs]
                        for occ in hgoccs:
                            assert link.genomes[occ.genomeIdx].species == 'Homo_sapiens', \
                                f"g: {occ.genomeIdx}, link: {link}"
                            if (occ.sequenceIdx == i) and (exonStart <= occ.position + link.span - 1) \
                                    and (occ.position < exonEnd):
                                #nhumanOccsThatHit += 1
                                humanOccsThatHit.add(occ)
                                # create new MultiLink with only the one human Occurrence 
                                #   to see how many unique links would hit
                                if link.singleProfile:
                                    exonHit = True
                                    mh = Links.MultiLink([occ]+remainingOccs, link.span, link.genomes, 
                                                         singleProfile=True)
                                    nlinksThatHit += mh.nUniqueLinks()
                                else:
                                    profileoccs = [occ] + [o for o in remainingOccs if o.profileIdx == occ.profileIdx]
                                    if len(profileoccs) > 1:
                                        exonHit = True
                                        mh = Links.MultiLink(profileoccs, link.span, link.genomes, singleProfile=True)
                                        nlinksThatHit += mh.nUniqueLinks()
                        
                if exonHit:
                    nexonsHit += 1

        # self.runIDs.append(runID)
        # self.motifs.append(motifs)
        # self.links.append(links)
        # self.nexonsHit.append(nexonsHit)
        # self.nlinks.append(nlinks)
        # self.nhumanExons.append(nhumanExons)
        # self.nlinksThatHit.append(nlinksThatHit)
        # self.trainingTime.append(time)
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
        # d = {
        #     'runIDs': self.runIDs,
        #     'nexonsHit': self.nexonsHit,
        #     'nhumanExons': self.nhumanExons,
        #     'nlinks': self.nlinks,
        #     'nlinksThatHit': self.nlinksThatHit,
        #     'trainingTime': self.trainingTime,
        #     'links': [[l.toDict() for l in links] for links in self.links],
        #     'motifs': [m.toDict() for m in self.motifs]
        # }

        # # for easier debugging
        # for key in d:
        #     try:
        #         _ = json.dumps(d[key])
        #     except Exception as e:
        #         logging.error(f"[MultiTrainingEvaluation.dump] >>> Error converting {key} to JSON, dump will fail.")
        #         logging.error(f"[MultiTrainingEvaluation.dump] >>> Exception:\n{e}")

        d = [t.toDict() for t in self.trainings]

        with open(filename, 'wt') as fh:
            json.dump(d, fh)

        # except Exception as e:
        #     logging.error(f"[MultiTrainingEvaluation.dump] >>> Error writing to file {filename}.")
        #     logging.error(f"[MultiTrainingEvaluation.dump] >>> Exception:\n{e}")
        #     logging.info(f"[MultiTrainingEvaluation.dump] >>> Writing data manually")
            
        #     with open(filename, 'wt') as fh:
        #         fh.write("{")
        #         fh.write('"runIDs": '+json.dumps(self.runIDs)+',\n')
        #         fh.write('"nexonsHit": '+json.dumps(self.nexonsHit)+',\n')
        #         fh.write('"nhumanExons": '+json.dumps(self.nhumanExons)+',\n')
        #         fh.write('"nlinks": '+json.dumps(self.nlinks)+',\n')
        #         fh.write('"nlinksThatHit": '+json.dumps(self.nlinksThatHit)+',\n')
        #         fh.write('"trainingTime": '+json.dumps(self.trainingTime)+',\n')
        #         fh.write('"links": [')
        #         links_inner = []
        #         for links in self.links:
        #             links_inner.append('[' + ', '.join([str(l.toDict()) for l in links]) + '],')
        #         fh.write(', '.join(links_inner))
        #         fh.write("],\n")
        #         fh.write('"motifs": [')
        #         fh.write(', '.join(str(m.toDict()) for m in self.motifs))
        #         fh.write("],\n")
        #         fh.write("}")



def loadMultiTrainingEvaluation(filename, allGenomes: list[sr.Genome], 
                                recalculate: bool = False) -> MultiTrainingEvaluation:
    """ Load a MultiTrainingEvaluation object from a JSON file. `allGenomes` must contain all Sequences (as a list of
    Genome objects) that were used in the training runs. 
    If `recalculate` is True, only the links and training time are loaded per training run and the metrics are 
    re-computed via MultiTrainingEvaluation.add_result(). This is useful if the metrics changed or were added later. """
    with open(filename, "rt") as fh:
        d = json.load(fh)

    seqIDsToGenomes = {allGenomes[gidx][sidx].id: (gidx, sidx) \
                           for gidx in range(len(allGenomes)) \
                           for sidx in range(len(allGenomes[gidx]))}

    mte = MultiTrainingEvaluation()
    for t in d:
        if len(t['links']) == 0:
            links = []
        else:
            # do we need this??
            subGenomes = []
            for genome in t['links'][0]['genomes']:
                subGenomes.append(sr.Genome())
                for seqID in genome:
                    assert seqID in seqIDsToGenomes, f"Sequence ID {seqID} not found in allGenomes."
                    gidx, sidx = seqIDsToGenomes[seqID]
                    #subGenomes[-1].append(allGenomes[gidx][sidx])
                    subGenomes[-1].addSequence(allGenomes[gidx][sidx])

            links = [Links.linkFromDict(l, subGenomes) for l in t['links']]

        if recalculate:
            mte.add_result(t['runID'],
                           motifs=loadMotifWrapperFromDict(t['motifs']),
                           links=links,
                           genomes=allGenomes,
                           time=t['trainingTime'])
        else:
            mte.trainings.append(
                TrainingEvaluation(runID=t['runID'], 
                                   motifs=loadMotifWrapperFromDict(t['motifs']),
                                   #links=[Links.linkFromDict(l, subGenomes) for l in t['links']],
                                   links=links,
                                   nexonsHit=t['nexonsHit'],
                                   nhumanExons=t['nhumanExons'],
                                   nhumanOccs=t['nhumanOccs'],
                                   nhumanOccsThatHit=t['nhumanOccsThatHit'],
                                   nlinks=t['nlinks'],
                                   nlinksThatHit=t['nlinksThatHit'],
                                   trainingTime=t['trainingTime'])
                )


    # mte.runIDs = d['runIDs']
    # mte.nexonsHit = d['nexonsHit']
    # mte.nhumanExons = d['nhumanExons']
    # mte.nlinks = d['nlinks']
    # mte.nlinksThatHit = d['nlinksThatHit']
    # mte.trainingTime = d['trainingTime']
    # mte.motifs = [loadMotifWrapperFromDict(m) for m in d['motifs']]

    # mte.links = []

    # seqIDsToGenomes = {allGenomes[gidx][sidx].id: (gidx, sidx) \
    #                        for gidx in range(len(allGenomes)) \
    #                        for sidx in range(len(allGenomes[gidx]))}
    # for links in d['links']:
    #     # in theory, we could have different genomes in different runs, but we assume that the genomes are the same
    #     subGenomes = []
    #     if len(links) == 0:
    #         mte.links.append([])
    #         continue

    #     for genome in links[0]['genomes']:
    #         subGenomes.append([])
    #         for seqID in genome:
    #             assert seqID in seqIDsToGenomes, f"Sequence ID {seqID} not found in allGenomes."
    #             gidx, sidx = seqIDsToGenomes[seqID]
    #             subGenomes[-1].append(allGenomes[gidx][sidx])

    #     mte.links.append([Links.linkFromDict(l, subGenomes) for l in links])

    return mte



def trainAndEvaluate(runID,
                     trainsetup: setup.ProfileFindingTrainingSetup, 
                     evaluator: MultiTrainingEvaluation,
                     outdir: str, outprefix: str = "",
                     trainingWithReporting: bool = True,
                     rand_seed: int = None) -> None: #-> tuple[float, float]:
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

    # build and randomly initialize profile model
    tf.keras.backend.clear_session() # avoid memory cluttering by remains of old models
    #specProModel = None
    specProModel = model.SpecificProfile(setup = trainsetup,
                                         alphabet_size = su.aa_alphabet_size, 
                                         rand_seed = rand_seed)

    # initialize profile model with seed profiles if not happened yet
    if trainsetup.initProfiles is None:
        P_logit_init = specProModel.seed_P_genome(trainsetup.data.genomes)
        specProModel.setP_logit(P_logit_init)

    # start training
    start = time()
    try:
        if trainingWithReporting:
            specProModel.train_reporting(verbose_freq=10)
        else:
            specProModel.train_classic(trainsetup.getDataset(repeat=True))

        end = time()
    except Exception as e:
        end = time()
        #print(f"Training failed after {end-start:.2f}.")
        #print("Exception:\n", e)
        logging.error(f"[training.trainAndEvaluate] >>> Training failed after {end-start:.2f}.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        
    training_time = end-start
    #print(f"Training time: {end-start:.2f}")
    logging.info(f"[training.trainAndEvaluate] >>> Training time: {training_time:.2f}")

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
        links, linkProfiles, skipped = Links.linksFromSites(sites, specProModel.setup.k*3, 
                                                            specProModel.setup.data.genomes, 1000)
        logging.debug(f"[training.trainAndEvaluate] >>> links[:{min(len(links), 2)}] {links[:min(len(links), 2)]}")
        logging.debug(f"[training.trainAndEvaluate] >>> len(links) {len(links)}")

        evaluator.add_result(runID, motifs, links, trainsetup.data.genomes, training_time)

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
        
        # # calculate accuracy
        # def accuracy(links: list[Links.Link]):
        #     assert trainsetup.data.genomes[0].species == 'Homo_sapiens'
        #     #nlinks = len(links)
        #     nlinks = 0
        #     nhumanExons = 0
        #     nexonsHit = 0
        #     nlinksThatHit = 0
        #     for link in links:
        #         occ = link.occs[0]
        #         if link.genomes[0].species == 'Homo_sapiens':
        #             nlinks += 1
            
        #     for sequence in trainsetup.data.genomes[0]:
        #         for i, exon in enumerate(sequence.genomic_elements):
        #             nhumanExons += 1
        #             exonHit = False
        #             exonStart, exonEnd = exon.getRelativePositions(sequence)
        #             for link in links:
        #                 occ = link.occs[0]
        #                 assert link.genomes[0].species == 'Homo_sapiens'
        #                 if (occ.sequenceIdx == i) and (exonStart <= occ.position + link.span - 1) and (occ.position < exonEnd):
        #                     exonHit = True
        #                     nlinksThatHit += 1
                            
        #             if exonHit:
        #                 nexonsHit += 1
                        
        #     sensitivity = nexonsHit/nhumanExons
        #     specificity = nlinksThatHit/nlinks
        #     #print(f"Sensitivity: {nexonsHit} / {nhumanExons} = {nexonsHit/nhumanExons}")
        #     #print(f"Specificity: {nlinksThatHit} / {nlinks} = {nlinksThatHit/nlinks}")
        #     logging.info("[training.trainAndEvaluate] >>> " + \
        #                  f"Sensitivity: {nexonsHit} / {nhumanExons} = {sensitivity}")
        #     logging.info("[training.trainAndEvaluate] >>> " + \
        #                  f"Specificity: {nlinksThatHit} / {nlinks} = {specificity}")
        #     return (sensitivity, specificity)
                        
        # return accuracy(links)

    except Exception as e:
        #print("Evaluation failed.")
        #print("Exception:\n", e)
        logging.error("[training.trainAndEvaluate] >>> Evaluation failed.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        #return None, None # return two Nones so caller can always unpack the result