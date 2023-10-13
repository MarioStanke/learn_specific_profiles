import logging
import matplotlib.pyplot as plt
import os
import plotly.express as px
import tensorflow as tf
from time import time

import Links
import model
import plotting
import ProfileFindingSetup as setup
import SequenceRepresentation as sr
import sequtils as su

def trainAndEvaluate(trainsetup: setup.ProfileFindingTrainingSetup, 
                     outdir: str, outprefix: str = "",
                     trainingWithReporting: bool = True,
                     rand_seed: int = None):
    """ 
    Train profiles on a given training setup and evaluate them. 
    Parameters:
        trainsetup: ProfileFindingTrainingSetup
            Training setup to use for training and evaluation.
        outdir: str
            Directory to save results to.
        outprefix: str
            Prefix for output files.
        trainingWithReporting: bool
            If True, training is done with reporting, otherwise with classic training.
        rand_seed: int
            Random seed for reproducibility.    
    """

    # build and randomly initialize profile model
    tf.keras.backend.clear_session() # avoid memory cluttering by remains of old models
    specProModel = None
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
    except Exception as e:
        end = time()
        #print(f"Training failed after {end-start:.2f}.")
        #print("Exception:\n", e)
        logging.error(f"[training.trainAndEvaluate] >>> Training failed after {end-start:.2f}.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")
        
    end = time()
    #print(f"Training time: {end-start:.2f}")
    logging.info(f"[training.trainAndEvaluate] >>> Training time: {end-start:.2f}")

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
        else:
            P, scores, losses = specProModel.getP_optimal(0, lossStatistics=True)

        thresh = Pthresh if trainingWithReporting else specProModel.setup.match_score_factor * scores

        onlyPid = None
        if onlyPid is None:
            sites, siteScores, _ = specProModel.get_profile_match_sites(specProModel.setup.getDataset(withPosTracking = True, original_data=True), 
                                                                        thresh, otherP = P)
        else:
            sites, siteScores, _ = specProModel.get_profile_match_sites(specProModel.setup.getDataset(withPosTracking = True, original_data=True), 
                                                                        thresh[onlyPid], otherP = P[:,:,onlyPid:onlyPid+1])
            
        #print("[DEBUG] >>> sites:", sites.numpy()[:20,]) # (sites, (genomeID, contigID, pos, u, f))
        logging.debug(f"[training.trainAndEvaluate] >>> sites: {sites.numpy()[:20,]}") # (sites, (genomeID, contigID, pos, u, f))
        links, linkProfiles, skipped = Links.linksFromSites(sites, specProModel.setup.k*3, specProModel.setup.data.genomes, 1000)
        #print(f"[DEBUG] >>> links[:{min(len(links), 2)}]", links[:min(len(links), 2)])
        #print("[DEBUG] >>> len(links)", len(links))
        logging.debug(f"[training.trainAndEvaluate] >>> links[:{min(len(links), 2)}] {links[:min(len(links), 2)]}")
        logging.debug(f"[training.trainAndEvaluate] >>> len(links) {len(links)}")

        kmerSites = []
        for kmer in trainsetup.initKmerPositions:
            kmerSites.extend(trainsetup.initKmerPositions[kmer])

        masksides = specProModel.getMaskedSites(0)

        plotting.drawGeneLinks_SequenceRepresentationData(trainsetup.data.genomes, links, 
                                                          imname=os.path.join(outdir, outprefix+"links.png"), 
                                                          kmerSites=kmerSites, kmerCol='deeppink',
                                                          maskingSites=masksides, maskingCol='chocolate')
        
        # calculate accuracy
        def accuracy(links: list[Links.Link]):
            assert trainsetup.data.genomes[0].species == 'Homo_sapiens'
            #nlinks = len(links)
            nlinks = 0
            nhumanExons = 0
            nexonsHit = 0
            nlinksThatHit = 0
            for link in links:
                occ = link.occs[0]
                if link.genomes[0].species == 'Homo_sapiens':
                    nlinks += 1
            
            for sequence in trainsetup.data.genomes[0]:
                for i, exon in enumerate(sequence.genomic_elements):
                    nhumanExons += 1
                    exonHit = False
                    exonStart, exonEnd = exon.getRelativePositions(sequence)
                    for link in links:
                        occ = link.occs[0]
                        assert link.genomes[0].species == 'Homo_sapiens'
                        if (occ.sequenceIdx == i) and (exonStart <= occ.position + link.span - 1) and (occ.position < exonEnd):
                            exonHit = True
                            nlinksThatHit += 1
                            
                    if exonHit:
                        nexonsHit += 1
                        
            #print(f"Sensitivity: {nexonsHit} / {nhumanExons} = {nexonsHit/nhumanExons}")
            #print(f"Specificity: {nlinksThatHit} / {nlinks} = {nlinksThatHit/nlinks}")
            logging.info("[training.trainAndEvaluate] >>> " + \
                         f"Sensitivity: {nexonsHit} / {nhumanExons} = {nexonsHit/nhumanExons}")
            logging.info("[training.trainAndEvaluate] >>> " + \
                         f"Specificity: {nlinksThatHit} / {nlinks} = {nlinksThatHit/nlinks}")
                        
        accuracy(links)

    except Exception as e:
        #print("Evaluation failed.")
        #print("Exception:\n", e)
        logging.error("[training.trainAndEvaluate] >>> Evaluation failed.")
        logging.error(f"[training.trainAndEvaluate] >>> Exception:\n{e}")