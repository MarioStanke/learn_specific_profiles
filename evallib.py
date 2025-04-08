from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import pandas as pd

from modules import plotting, training, utils
from modules import SequenceRepresentation as sr

def load_mast(path: Path, k: int = None):
    """ Load MAST output from a best_hits file and return a DataFrame with the relevant columns 
    [sequence, hit_start, hit_end, hit_len, hit_center] as zero-based, end exclusive coordinates.
    Parameter k is optional to check that the length of the hits in the MAST output is consistent with k. """
    # if file is empty, return an empty DataFrame
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=['sequence', 'hit_start', 'hit_end', 'hit_len', 'hit_center'])
    
    mast = pd.read_csv(str(path), sep="\s+",
                       names=["sequence", "(strand+/-)motif id", "alt_id", "-", "hit_start", "hit_end", "score", 
                               "hit_p-value"],
                       comment="#")
    # coordinates seem to be 1-based, and the end is inclusive (i.e. [start, end], not [start, end))
    # we want 0-based, end-exclusive coordinates, so we subtract 1 from the start and should be good
    mast['hit_start'] = mast['hit_start']-1
    assert (mast['hit_start'] >= 0).all(), f"[load_mast] encountered negative starts: {mast['hit_start'].min()}"
    assert (mast['hit_end'] > mast['hit_start']).all(), \
        f"[load_mast] some end points before start: {mast['hit_end'].min()} < {mast['hit_start'].min()}"
    mast['hit_len'] = mast['hit_end']-mast['hit_start']
    if k is not None:
        # mast allows for hits of not exactly k length, so we also allow a bit of leeway in both directions
        d=2
        assert ((mast['hit_len'] >= k-2) & (mast['hit_len'] <= k+2)).all(), \
            f"[load_mast] inconsistent hit lengths (require {k=}+-{d}): {mast['hit_len'].value_counts()}"
    mast['hit_center'] = mast['hit_start']+(mast['hit_len']//2)

    return mast[['sequence', 'hit_start', 'hit_end', 'hit_len', 'hit_center']]


def load_fimo(path: Path, k: int = None):
    """ Load FIMO output from a best_site.narrowPeak file and return a DataFrame with the relevant columns
    [sequence, hit_start, hit_end, hit_len, hit_center].
    Parameter k is optional to check that the length of the hits in the FIMO output is consistent with k. """
    # if file is empty, return an empty DataFrame
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=['sequence', 'hit_start', 'hit_end', 'hit_len', 'hit_center'])
    
    fimo = utils.readBEDFile(Path(str(path)))
    assert (fimo.columns == ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 
                            'signalValue', 'pValue', 'qValue', 'peak']).all(), \
        f"[load_fimo] unexpected columns: {fimo.columns}"
    fimo.columns = ['sequence', 'hit_start', 'hit_end'] + fimo.columns[3:].tolist()
    assert (fimo['hit_start'] >= 0).all(), f"[load_fimo] encountered negative starts: {fimo['hit_start'].min()}"
    assert (fimo['hit_end'] > fimo['hit_start']).all(), \
        f"[load_fimo] some end points before start: {fimo['hit_end'].min()} < {fimo['hit_start'].min()}"
    assert (fimo['peak'] >= 0).all(), f"[load_fimo] negative peak offsets: {fimo['peak'].min()}"
    assert (fimo['peak'] < fimo['hit_end']).all(), \
        f"[load_fimo] peak offsets greater than hit end: {fimo[['hit_end', 'peak']]}"
    fimo['hit_len'] = fimo['hit_end']-fimo['hit_start']
    if k is not None:
        # fimo allows for hits of not exactly k length, so we also allow a bit of leeway in both directions
        d=2
        assert ((fimo['hit_len'] >= k-2) & (fimo['hit_len'] <= k+2)).all(), \
            f"[load_fimo] inconsistent hit lengths (require {k=}+-{d}): {fimo['hit_len'].value_counts()}"
    fimo['hit_center'] = fimo['hit_start'] + fimo['peak']
    
    return fimo[['sequence', 'hit_start', 'hit_end', 'hit_len', 'hit_center']]


def evaluate_experiment(ed: Path, 
                        eval_file: str = "evaluator_test.json", 
                        neg_eval_file: str = "evaluator_negative_test.json", 
                        streme_eval_file: str = "STREME/streme_evaluator_dummymodel_test.json", 
                        streme_neg_eval_file: str = "STREME/streme_evaluator_dummymodel_negative_test.json",
                        fimo_sites_file: str = "fimo/best_site.narrowPeak",
                        fimo_neg_sites_file: str = "fimo/neg/best_site.narrowPeak",
                        fimo_streme_sites_file: str = "fimo/STREME/best_site.narrowPeak",
                        fimo_streme_neg_sites_file: str = "fimo/neg/STREME/best_site.narrowPeak",
                        mast_sites_file: str = "mast/best_hits.tsv",
                        mast_neg_sites_file: str = "mast/best_hits_neg.tsv",
                        mast_streme_sites_file: str = "mast/STREME/best_hits.tsv",
                        mast_streme_neg_sites_file: str = "mast/STREME/best_hits_neg.tsv",
                        training_seq_file: str = "training_sequences_0.json", 
                        test_seq_file: str = "test_sequences_0.json",
                        neg_test_seq_file: str = "negative_test_sequences_0.json",
                        settings_file: str = "settings.json",
                        show_plots: bool = True, silent: bool = False):
    assert ed.is_dir(), f"{ed} is not a directory"
    assert (ed / eval_file).is_file(), f"{ed / eval_file} does not exist"
    assert (ed / neg_eval_file).is_file(), f"{ed / neg_eval_file} does not exist"
    assert (ed / streme_eval_file).is_file(), f"{ed / streme_eval_file} does not exist"
    assert (ed / streme_neg_eval_file).is_file(), f"{ed / streme_neg_eval_file} does not exist"
    assert (ed / fimo_sites_file).is_file(), f"{ed / fimo_sites_file} does not exist"
    assert (ed / fimo_neg_sites_file).is_file(), f"{ed / fimo_neg_sites_file} does not exist"
    assert (ed / fimo_streme_sites_file).is_file(), f"{ed / fimo_streme_sites_file} does not exist"
    assert (ed / fimo_streme_neg_sites_file).is_file(), f"{ed / fimo_streme_neg_sites_file} does not exist"
    assert (ed / mast_sites_file).is_file(), f"{ed / mast_sites_file} does not exist"
    assert (ed / mast_neg_sites_file).is_file(), f"{ed / mast_neg_sites_file} does not exist"
    assert (ed / mast_streme_sites_file).is_file(), f"{ed / mast_streme_sites_file} does not exist"
    assert (ed / mast_streme_neg_sites_file).is_file(), f"{ed / mast_streme_neg_sites_file} does not exist"
    assert (ed / training_seq_file).is_file(), f"{ed / training_seq_file} does not exist"
    assert (ed / test_seq_file).is_file(), f"{ed / test_seq_file} does not exist"
    assert (ed / neg_test_seq_file).is_file(), f"{ed / neg_test_seq_file} does not exist"
    assert (ed / settings_file).is_file(), f"{ed / settings_file} does not exist"

    with open(ed / settings_file, "r") as f:
        settings = json.load(f)
    training_genomes = sr.loadJSONGenomeList(str(ed / training_seq_file))
    test_genomes = sr.loadJSONGenomeList(str(ed / test_seq_file))
    neg_test_genomes = sr.loadJSONGenomeList(str(ed / neg_test_seq_file))
    evaluator = training.loadMultiTrainingEvaluation(str(ed / eval_file), test_genomes)
    neg_evaluator = training.loadMultiTrainingEvaluation(str(ed / neg_eval_file), neg_test_genomes)
    streme_evaluator = training.loadMultiTrainingEvaluation(str(ed / streme_eval_file), test_genomes)
    streme_neg_evaluator = training.loadMultiTrainingEvaluation(str(ed / streme_neg_eval_file), neg_test_genomes)
    assert len(evaluator.trainings) == 1, f"expected 1 training, got {len(evaluator.trainings)}"
    assert len(neg_evaluator.trainings) == 1, f"expected 1 training, got {len(neg_evaluator.trainings)}"
    assert len(streme_evaluator.trainings) == 1, f"expected 1 training, got {len(streme_evaluator.trainings)}"
    assert len(streme_neg_evaluator.trainings) == 1, f"expected 1 training, got {len(streme_neg_evaluator.trainings)}"
    fimo_sites = load_fimo(ed / fimo_sites_file, settings['k'])
    fimo_neg_sites = load_fimo(ed / fimo_neg_sites_file, settings['k'])
    fimo_streme_sites = load_fimo(ed / fimo_streme_sites_file, settings['k'])
    fimo_streme_neg_sites = load_fimo(ed / fimo_streme_neg_sites_file, settings['k'])
    mast_sites = load_mast(ed / mast_sites_file, settings['k'])
    mast_neg_sites = load_mast(ed / mast_neg_sites_file, settings['k'])
    mast_streme_sites = load_mast(ed / mast_streme_sites_file, settings['k'])
    mast_streme_neg_sites = load_mast(ed / mast_streme_neg_sites_file, settings['k'])

    ref_types = set()
    for genomes in [training_genomes, test_genomes, neg_test_genomes]:
        for genome in genomes:
            for seq in genome:
                assert seq.elementsPossible(), f"no elements possible in {seq}"
                ref_types.update([e.type for e in seq.genomic_elements])

    def _evalHits(hits: list[training.Links.MultiLink], genomes: list[sr.Genome]):
        all_refs: dict[str, list[tuple[float, str]]] = {
            rt: [] for rt in ref_types
        }
        sidToIdcs: dict[str, tuple[int, int]] = {}
        for gid, genome in enumerate(genomes):
            for sid, seq in enumerate(genome):
                if seq.id not in sidToIdcs:
                    sidToIdcs[seq.id] = (gid, sid)
                elif not silent:
                    logging.warning(f"sequence id {seq.id} occurs multiple times")
                    
                assert seq.elementsPossible(), f"no elements in {seq}"
                for element in seq.genomic_elements:
                    ref, _ = element.getRelativePositions(seq)
                    rel_ref = 100*ref/len(seq)
                    all_refs[element.type].append((rel_ref, seq.id))

        ref_hit_distances: dict[str, list[tuple[int, tuple[int, str]]]] = {rt: [] for rt in ref_types}
        relative_hits: list[tuple[float, str]] = []
        for link in hits:
            for occs in link.occs:
                if len(occs) > 0:
                    assert occs[0].sequence.id in sidToIdcs, f"sequence id {occs[0].sequence.id} not found"
                    assert all([occs[0].sequence.id == o.sequence.id for o in occs]), \
                        "all occurrences must be in the same sequence"
                    
                    gid, sid = sidToIdcs[occs[0].sequence.id]
                    refs: dict[str, list[int]] = {rt: [] for rt in ref_types}
                    for element in genomes[gid].sequences[sid].genomic_elements:
                        ref, _ = element.getRelativePositions(genomes[gid][sid])
                        refs[element.type].append(ref)

                    for occ in occs:
                        hit_start = occ.position
                        hit_len = occ.sitelen
                        hit_end = hit_start + hit_len
                        hit_center = hit_start + hit_len // 2
                        relative_hits.append((100*hit_center/len(genomes[gid][sid]), occ.sequence.id))

                        for rt in refs.keys():
                            if len(refs[rt]) > 0:
                                ref_distances = []
                                for ref in refs[rt]:
                                    dist = 0 if hit_start <= ref < hit_end else min(abs(ref - hit_start), 
                                                                                    abs(ref - hit_end))
                                    ref_distances.append(dist)

                                i, min_d = min(enumerate(ref_distances), key=lambda x: x[1]) # argmin & min in one go
                                ref_hit_distances[rt].append((min_d, (refs[rt][i], occ.sequence.id))) # dist, (ref, seq)

        return all_refs, ref_hit_distances, relative_hits
    

    def _evalEvaluator(evaluator: training.MultiTrainingEvaluation, genomes: list[sr.Genome]):
        return _evalHits(evaluator.trainings[0].links, genomes)
    
    def _evalSites(sites: pd.DataFrame, genomes: list[sr.Genome]):
        assert sites.columns.tolist() == ['sequence', 'hit_start', 'hit_end', 'hit_len', 'hit_center'], \
            f"unexpected columns {sites.columns.tolist()}"
        # create multilinks from sites
        seqidToIdcs = {}
        for gid, genome in enumerate(genomes):
            for sid, seq in enumerate(genome):
                if seq.id not in seqidToIdcs:
                    seqidToIdcs[seq.id] = (gid, sid)
                elif not silent:
                    logging.warning(f"sequence id {seq.chromosome} occurs multiple times")
                
        assert all([sid in seqidToIdcs for sid in sites['sequence']]), \
            f"{len([s for s in sites['sequence'] if s not in seqidToIdcs])}/{len(set(sites['sequence']))} sequence ids {sorted(set([s for s in sites['sequence'] if s not in seqidToIdcs]))}\n\n" \
                + f"not found in {len(seqidToIdcs.keys())} genomes {sorted(seqidToIdcs.keys())}"
        occs = [training.Links.Occurrence(sequence=genomes[seqidToIdcs[seqid][0]][seqidToIdcs[seqid][1]], 
                                          position=hit_start, 
                                          strand="+", # ignoring strand information for now
                                          sitelen=hit_len) \
                for seqid, hit_start, hit_end, hit_len, hit_center in sites.itertuples(index=False)]
        links = [training.Links.MultiLink(occs, singleProfile=True)] # assume this, must be changed if we want to distinguish single profiles
        return _evalHits(links, genomes)

    # get evaluation results
    all_refs, ref_hit_distances, relative_hits = _evalEvaluator(evaluator, test_genomes)
    _, ref_hit_distances_streme, relative_hits_streme = _evalEvaluator(streme_evaluator, test_genomes)
    _, ref_hit_distances_fimo, relative_hits_fimo = _evalSites(fimo_sites, test_genomes)
    _, ref_hit_distances_fimo_streme, relative_hits_fimo_streme = _evalSites(fimo_streme_sites, test_genomes)
    _, ref_hit_distances_mast, relative_hits_mast = _evalSites(mast_sites, test_genomes)
    _, ref_hit_distances_mast_streme, relative_hits_mast_streme = _evalSites(mast_streme_sites, test_genomes)

    all_refs_neg, ref_hit_distances_neg, relative_hits_neg = _evalEvaluator(neg_evaluator, neg_test_genomes)
    assert all([len(all_refs_neg[rt]) == 0 for rt in all_refs_neg.keys()]), \
        "no reference sites should be found in negative test sequences"
    assert all([len(ref_hit_distances_neg[rt]) == 0 for rt in ref_hit_distances_neg.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"
    _, ref_hit_distances_neg_streme, relative_hits_neg_streme = _evalEvaluator(streme_neg_evaluator, neg_test_genomes)
    assert all([len(ref_hit_distances_neg_streme[rt]) == 0 for rt in ref_hit_distances_neg_streme.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"
    _, ref_hit_distances_fimo_neg, relative_hits_fimo_neg = _evalSites(fimo_neg_sites, neg_test_genomes)
    _, ref_hit_distances_fimo_streme_neg, relative_hits_fimo_streme_neg = _evalSites(fimo_streme_neg_sites, 
                                                                                     neg_test_genomes)
    _, ref_hit_distances_mast_neg, relative_hits_mast_neg = _evalSites(mast_neg_sites, neg_test_genomes)
    _, ref_hit_distances_mast_streme_neg, relative_hits_mast_streme_neg = _evalSites(mast_streme_neg_sites, 
                                                                                     neg_test_genomes)
    assert all([len(ref_hit_distances_fimo_neg[rt]) == 0 for rt in ref_hit_distances_fimo_neg.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"
    assert all([len(ref_hit_distances_fimo_streme_neg[rt]) == 0 for rt in ref_hit_distances_fimo_streme_neg.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"
    assert all([len(ref_hit_distances_mast_neg[rt]) == 0 for rt in ref_hit_distances_mast_neg.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"
    assert all([len(ref_hit_distances_mast_streme_neg[rt]) == 0 for rt in ref_hit_distances_mast_streme_neg.keys()]), \
        "no reference site - hit distances should be found in negative test sequences"

    if show_plots:
        fig = plotting.ownPlotlyHist({rt: [t[0] for t in all_refs[rt]] for rt in all_refs})
        fig.update_layout(title="Reference site distribution in test sequences", 
                          xaxis_title="relative position * 100", yaxis_title="reference site count")
        fig.show()

        fig = plotting.ownPlotlyHist({f"{sites} {rt}": [t[0] for t in dists[rt]] \
                                        for sites, dists in {'ProfileFinding': ref_hit_distances, 
                                                             'FIMO': ref_hit_distances_fimo, 
                                                             'MAST': ref_hit_distances_mast}.items() \
                                            for rt in dists}, 
                                     rel=True)
        fig.update_layout(title="Distance of hits to reference sites in test sequences", 
                          xaxis_title="distance to closest reference site", yaxis_title="relative frequency")
        fig.show()

        fig = plotting.ownPlotlyHist({f"{sites} {rt}": [t[0] for t in dists[rt]] \
                                        for sites, dists in {'ProfileFinding STREME': ref_hit_distances_streme, 
                                                             'FIMO STREME': ref_hit_distances_fimo_streme, 
                                                             'MAST STREME': ref_hit_distances_mast_streme}.items() \
                                            for rt in dists}, 
                                     rel=True)
        fig.update_layout(title="Distance of hits to reference sites in test sequences | STREME", 
                          xaxis_title="distance to closest reference site", yaxis_title="relative frequency")
        fig.show()

        fig = plotting.ownPlotlyHist({f"relative hits {mode}": [t[0] for t in hits] \
                                      for mode, hits in {'ProfileFinding': relative_hits,
                                                         'FIMO': relative_hits_fimo,
                                                         'MAST': relative_hits_mast}.items()})
        fig.update_layout(title="Relative hit positions in test sequences", 
                          xaxis_title="relative position * 100", yaxis_title="hit count")
        fig.show()

        fig = plotting.ownPlotlyHist({f"relative hits {mode}": [t[0] for t in hits] \
                                      for mode, hits in {'ProfileFinding STREME': relative_hits_streme,
                                                         'FIMO STREME': relative_hits_fimo_streme,
                                                         'MAST STREME': relative_hits_mast_streme}.items()})
        fig.update_layout(title="Relative hit positions in test sequences | STREME", 
                          xaxis_title="relative position * 100", yaxis_title="hit count")
        fig.show()

        fig = plotting.ownPlotlyHist({f"relative hits {mode}": [t[0] for t in hits] \
                                      for mode, hits in {'ProfileFinding negative': relative_hits_neg,
                                                         'FIMO negative': relative_hits_fimo_neg,
                                                         'MAST negative': relative_hits_mast_neg}.items()})
        fig.update_layout(title="Relative hit positions in negative sequences", 
                          xaxis_title="relative position * 100", yaxis_title="hit count")
        fig.show()

        fig = plotting.ownPlotlyHist({f"relative hits {mode}": [t[0] for t in hits] \
                                      for mode, hits in {'ProfileFinding negative STREME': relative_hits_neg_streme,
                                                         'FIMO negative STREME': relative_hits_fimo_streme_neg,
                                                         'MAST negative STREME': relative_hits_mast_streme_neg}.items()}
                                                         )
        fig.update_layout(title="Relative hit positions in negative sequences | STREME", 
                          xaxis_title="relative position * 100", yaxis_title="hit count")
        fig.show()

    @dataclass
    class ModalityEval:
        model: str # one of 'ProfileFinding' or 'STREME'
        hitsrc: str # one of 'pf_model', 'FIMO', 'MAST'
        ref_hit_distances: dict[str, list[tuple[int, tuple[int, str]]]]
        relative_hits: list[tuple[float, str]]
        relative_hits_neg: list[tuple[float, str]]

    @dataclass
    class EvalResult:
        genomes: list[sr.Genome]
        neg_genomes: list[sr.Genome]
        all_refs: dict[str, list[tuple[float, str]]]
        eval_profile_finding: ModalityEval
        eval_profile_finding_fimo: ModalityEval
        eval_profile_finding_mast: ModalityEval
        eval_streme: ModalityEval
        eval_streme_fimo: ModalityEval
        eval_streme_mast: ModalityEval

    return EvalResult(test_genomes, neg_test_genomes, all_refs,
                      eval_profile_finding=ModalityEval('ProfileFinding', 'pf_model', ref_hit_distances, 
                                                        relative_hits, relative_hits_neg),
                      eval_profile_finding_fimo=ModalityEval('ProfileFinding', 'FIMO', ref_hit_distances_fimo, 
                                                             relative_hits_fimo, relative_hits_fimo_neg),
                      eval_profile_finding_mast=ModalityEval('ProfileFinding', 'MAST', ref_hit_distances_mast, 
                                                             relative_hits_mast, relative_hits_mast_neg),
                      eval_streme=ModalityEval('STREME', 'pf_model', ref_hit_distances_streme, 
                                               relative_hits_streme, relative_hits_neg_streme),
                      eval_streme_fimo=ModalityEval('STREME', 'FIMO', ref_hit_distances_fimo_streme, 
                                                    relative_hits_fimo_streme, relative_hits_fimo_streme_neg),
                      eval_streme_mast=ModalityEval('STREME', 'MAST', ref_hit_distances_mast_streme, 
                                                    relative_hits_mast_streme, relative_hits_mast_streme_neg))


def evaluate_multiple_experiments(experiment_dirs: list[Path], reevaluate = False):
    sequences: set[str] = set()
    negative_sequences: set[str] = set()
    refsites: dict[str, list[float]] = {}
    sequences_with_refsites: dict[str, set[str]] = {}

    @dataclass
    class ModalityEval:
        model: str # one of 'ProfileFinding' or 'STREME'
        hitsrc: str # one of 'ProfileFinding', 'FIMO', 'MAST'
        hits: list[float] = field(default_factory=list)
        negative_hits: list[float] = field(default_factory=list)
        sequences_with_hits: set[str] = field(default_factory=set)
        negative_sequences_with_hits: set[str] = field(default_factory=set)
        hits_on_refsites: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
        ref_distances: dict[str, list[int]] = field(default_factory=dict)
        sequences_with_hits_on_refsites: dict[str, set[str]] = field(default_factory=dict)

        def to_json_dict(self):
            return {
                "model": self.model,
                "hitsrc": self.hitsrc,
                "hits": self.hits,
                "negative_hits": self.negative_hits,
                "sequences_with_hits": list(self.sequences_with_hits),
                "negative_sequences_with_hits": list(self.negative_sequences_with_hits),
                "hits_on_refsites": {rt: [list(t) for t in dists] for rt, dists in self.hits_on_refsites.items()},
                "ref_distances": self.ref_distances,
                "sequences_with_hits_on_refsites": {rt: list(rs) for rt, rs in self.sequences_with_hits_on_refsites.items()}
            }
        
        @classmethod
        def from_json_dict(cls, data: dict):
            instance = cls(data['model'], data['hitsrc'])
            instance.hits = data['hits']
            instance.negative_hits = data['negative_hits']
            instance.sequences_with_hits = set(data['sequences_with_hits'])
            instance.negative_sequences_with_hits = set(data['negative_sequences_with_hits'])
            instance.hits_on_refsites = {rt: [tuple(t) for t in dists] for rt, dists in data['hits_on_refsites'].items()}
            instance.ref_distances = data['ref_distances']
            instance.sequences_with_hits_on_refsites = {rt: set(rs) for rt, rs in data['sequences_with_hits_on_refsites'].items()}
            return instance

        def getStats(self, sequences: set[str], negative_sequences: set[str], refsites: dict[str, list[float]]):
            statsdict = {
                "number_of_hits_test": len(self.hits),
                "number_of_hits_neg": len(self.negative_hits),
                "number_of_hits_test_per_seq": len(self.hits)/len(sequences),
                "number_of_hits_neg_per_seq": len(self.negative_hits)/len(negative_sequences),
                "test_sequences_with_geq_1_hit": 100*len(self.sequences_with_hits)/len(sequences),
                "neg_sequences_with_geq_1_hit": 100*len(self.negative_sequences_with_hits)/len(negative_sequences),
            }
            for rt in refsites.keys():
                # avoid key errors, although unlikely to happen
                if rt not in self.sequences_with_hits_on_refsites:
                    self.sequences_with_hits_on_refsites[rt] = set()
                if rt not in self.hits_on_refsites:
                    self.hits_on_refsites[rt] = []
                if rt not in self.ref_distances:
                    self.ref_distances[rt] = []

                statsdict[rt] = {
                    "refsites_hit": 100*len(set(self.hits_on_refsites[rt]))/len(refsites[rt]),
                    "sequences_with_hits_on_refsites": 100*len(self.sequences_with_hits_on_refsites[rt])/len(sequences)}

            return statsdict
        
        def printStats(self, sequences: set[str], negative_sequences: set[str], refsites: dict[str, list[float]]):
            stats = self.getStats(sequences, negative_sequences, refsites)
            print(f"""
[Model {self.model} | Hits from {self.hitsrc}]
-------{'-'*len(self.model)}-------------{'-'*len(self.hitsrc)}-

Number of hits (test): {stats['number_of_hits_test']} | {stats['number_of_hits_test_per_seq']:.2f} hits per seq
Number of hits (neg.): {stats['number_of_hits_neg']} | {stats['number_of_hits_neg_per_seq']:.2f} hits per seq

---

Test sequences with >= 1 hit: {stats['test_sequences_with_geq_1_hit']:.2f}% ({len(self.sequences_with_hits)}/{len(sequences)})
Neg. sequences with >= 1 hit: {stats['neg_sequences_with_geq_1_hit']:.2f}% ({len(self.negative_sequences_with_hits)}/{len(negative_sequences)})""")
    
            for rt in refsites.keys():
                print(f"""
[Reference sites: {rt}]:
    Reference sites hit: {stats['refsites_hit']:.2f}% ({len(set(self.hits_on_refsites[rt]))})
    Sequences with hits on reference sites: {stats['sequences_with_hits_on_refsites']:.2f}% ({len(self.sequences_with_hits_on_refsites[rt])})""")


    # return result to avoid re-running the whole thing when just a new plot or so is needed
    @dataclass
    class EvalResult:
        sequences: set[str]
        negative_sequences: set[str]
        refsites: dict[str, list[float]]
        sequences_with_refsites: dict[str, set[str]]
        eval_ProfileFinding: ModalityEval
        eval_ProfileFinding_FIMO: ModalityEval
        eval_ProfileFinding_MAST: ModalityEval
        eval_STREME: ModalityEval
        eval_STREME_FIMO: ModalityEval
        eval_STREME_MAST: ModalityEval

        def to_json(self, path: Path):
            with open(path, 'w') as f:
                json.dump({"sequences": list(self.sequences),
                           "negative_sequences": list(self.negative_sequences),
                           "refsites": self.refsites,
                           "sequences_with_refsites": {rt: list(rs) for rt, rs in self.sequences_with_refsites.items()},
                           "eval_ProfileFinding": self.eval_ProfileFinding.to_json_dict(),
                           "eval_ProfileFinding_FIMO": self.eval_ProfileFinding_FIMO.to_json_dict(),
                           "eval_ProfileFinding_MAST": self.eval_ProfileFinding_MAST.to_json_dict(),
                           "eval_STREME": self.eval_STREME.to_json_dict(),
                           "eval_STREME_FIMO": self.eval_STREME_FIMO.to_json_dict(),
                           "eval_STREME_MAST": self.eval_STREME_MAST.to_json_dict()}, f, indent=4)
                
        @classmethod
        def from_json(cls, path: Path):
            with open(path, 'r') as f:
                data = json.load(f)
            instance = cls(set(data['sequences']), 
                           set(data['negative_sequences']), 
                           data['refsites'],
                           {rt: set(rs) for rt, rs in data['sequences_with_refsites'].items()},
                           ModalityEval.from_json_dict(data['eval_ProfileFinding']),
                           ModalityEval.from_json_dict(data['eval_ProfileFinding_FIMO']),
                           ModalityEval.from_json_dict(data['eval_ProfileFinding_MAST']),
                           ModalityEval.from_json_dict(data['eval_STREME']),
                           ModalityEval.from_json_dict(data['eval_STREME_FIMO']),
                           ModalityEval.from_json_dict(data['eval_STREME_MAST']))
            return instance

        def printStats(self):
            print(f"""
Number of test sequences: {len(self.sequences)}
Number of test sequences with reference sites: {[str(rt)+' '+str(len(self.sequences_with_refsites[rt])) for rt in self.sequences_with_refsites]}
Number of neg. sequences: {len(self.negative_sequences)}""")
            for rt in self.refsites.keys():
                print(f"""
[Reference sites: {rt}]:
    Number of reference sites: {len(self.refsites[rt])}
    Sequences with reference sites: {100*len(self.sequences_with_refsites[rt])/len(self.sequences):.2f}% ({len(self.sequences_with_refsites[rt])})
    Reference sites per sequence: {len(self.refsites[rt])/len(self.sequences):.2f} (all) | {len(self.refsites[rt])/len(self.sequences_with_refsites[rt]):.2f} (seq. w/ ref. sites)""")
                
            self.eval_ProfileFinding.printStats(self.sequences, self.negative_sequences, self.refsites)
            self.eval_ProfileFinding_FIMO.printStats(self.sequences, self.negative_sequences, self.refsites)
            self.eval_ProfileFinding_MAST.printStats(self.sequences, self.negative_sequences, self.refsites)
            self.eval_STREME.printStats(self.sequences, self.negative_sequences, self.refsites)
            self.eval_STREME_FIMO.printStats(self.sequences, self.negative_sequences, self.refsites)
            self.eval_STREME_MAST.printStats(self.sequences, self.negative_sequences, self.refsites)

        def plots(self):
            fig = plotting.ownPlotlyHist({f"{e.model} - {e.hitsrc} hits": e.hits \
                                          for e in [self.eval_ProfileFinding, self.eval_ProfileFinding_FIMO, 
                                                    self.eval_ProfileFinding_MAST, self.eval_STREME, 
                                                    self.eval_STREME_FIMO, self.eval_STREME_MAST]})
            fig.update_layout(title="Relative hit positions in test sequences", 
                            xaxis_title="relative position * 100", yaxis_title="hit count")
            fig.show()

            fig = plotting.ownPlotlyHist({f"{e.model} - {e.hitsrc} hits": e.negative_hits \
                                          for e in [self.eval_ProfileFinding, self.eval_ProfileFinding_FIMO, 
                                                    self.eval_ProfileFinding_MAST, self.eval_STREME, 
                                                    self.eval_STREME_FIMO, self.eval_STREME_MAST]})
            fig.update_layout(title="Relative hit positions in negative test sequences", 
                            xaxis_title="relative position * 100", yaxis_title="hit count")
            fig.show()

            fig = plotting.ownPlotlyHist({
                f"{e.model} - {e.hitsrc} <-> {rt}": e.ref_distances[rt] \
                    for e in [self.eval_ProfileFinding, self.eval_ProfileFinding_FIMO, self.eval_ProfileFinding_MAST, 
                              self.eval_STREME, self.eval_STREME_FIMO, self.eval_STREME_MAST] \
                    for rt in e.ref_distances.keys()
            }, rel=True)
            fig.update_layout(title="Distance of hits <-> reference sites in test sequences", 
                            xaxis_title="distance to closest reference site", yaxis_title="relative frequency")
            fig.show()

            fig = plotting.ownPlotlyHist(self.refsites)
            fig.update_layout(title="Reference site distribution in test sequences", 
                            xaxis_title="relative position * 100", yaxis_title="site count")
            fig.show()

    result_file = Path(experiment_dirs[0].parent, "full_evaluation.json")
    if result_file.is_file() and not reevaluate:
        logging.info(f"Loading results from {result_file} instead of re-evaluating")
        return EvalResult.from_json(result_file)
                

    # --- evaluation ---

    eval_ProfileFinding: ModalityEval = ModalityEval('ProfileFinding', 'pf_model')
    eval_ProfileFinding_FIMO: ModalityEval = ModalityEval('ProfileFinding', 'FIMO')
    eval_ProfileFinding_MAST: ModalityEval = ModalityEval('ProfileFinding', 'MAST')
    eval_STREME: ModalityEval = ModalityEval('STREME', 'pf_model')
    eval_STREME_FIMO: ModalityEval = ModalityEval('STREME', 'FIMO')
    eval_STREME_MAST: ModalityEval = ModalityEval('STREME', 'MAST')

    for ed in experiment_dirs:
        try:
            result = evaluate_experiment(ed, show_plots=False, silent=True)
        except Exception as e:
            print(f"Could not evaluate experiment {ed}:\n\t{e}")
            continue

        sequences.update([seq.id for genome in result.genomes for seq in genome])
        negative_sequences.update([seq.id for genome in result.neg_genomes for seq in genome])
        for rt in result.all_refs.keys():
            if rt not in refsites:
                refsites[rt] = []
            if rt not in sequences_with_refsites:
                sequences_with_refsites[rt] = set()

            refsites[rt].extend([t[0] for t in result.all_refs[rt]])
            sequences_with_refsites[rt].update([t[1] for t in result.all_refs[rt]])

        for r, e in zip([result.eval_profile_finding, result.eval_profile_finding_fimo, result.eval_profile_finding_mast,
                         result.eval_streme, result.eval_streme_fimo, result.eval_streme_mast],
                        [eval_ProfileFinding, eval_ProfileFinding_FIMO, eval_ProfileFinding_MAST,
                         eval_STREME, eval_STREME_FIMO, eval_STREME_MAST]):
            assert r.model == e.model
            assert r.hitsrc == e.hitsrc
                
            e.hits.extend([t[0] for t in r.relative_hits])
            e.negative_hits.extend([t[0] for t in r.relative_hits_neg])
            e.sequences_with_hits.update([t[1] for t in r.relative_hits])
            e.negative_sequences_with_hits.update([t[1] for t in r.relative_hits_neg])
            for rt in r.ref_hit_distances.keys():
                if rt not in e.hits_on_refsites:
                    e.hits_on_refsites[rt] = []
                if rt not in e.sequences_with_hits_on_refsites:
                    e.sequences_with_hits_on_refsites[rt] = set()
                if rt not in e.ref_distances:
                    e.ref_distances[rt] = []

                e.ref_distances[rt].extend([t[0] for t in r.ref_hit_distances[rt]])
                refsites_hit = [t[1] for t in r.ref_hit_distances[rt] if t[0] == 0] # distance 0, t[1] is (refsite, seq)
                e.hits_on_refsites[rt].extend(refsites_hit)
                e.sequences_with_hits_on_refsites[rt].update([t[1] for t in refsites_hit]) # t[1] is seq

    full_eval = EvalResult(sequences, negative_sequences, refsites, sequences_with_refsites,
                           eval_ProfileFinding, eval_ProfileFinding_FIMO, eval_ProfileFinding_MAST,
                           eval_STREME, eval_STREME_FIMO, eval_STREME_MAST)
    
    # store result in json file to avoid time-consuming re-evaluation
    full_eval.to_json(result_file)

    return full_eval


def compare_runs(run_dirs: dict[str, list[Path]], reevaluate = False):
    """
    Compare multiple runs, input is a dict of a descriptive run name and the corresponding base directories.
    """
    # model: str # one of 'ProfileFinding' or 'STREME'
    # hitsrc: str # one of 'ProfileFinding', 'FIMO', 'MAST'

    comp = {
        hitsrc: {
            "run": [],
            "sequences": [],
            "ref. sites (min, max)": [],
            "ProfileFinding hits": [],
            "ProfileFinding hits (-)": [],
            "STREME hits": [],
            "STREME hits (-)": [],
            "ProfileFinding hit seqs": [],
            "ProfileFinding hit seqs (-)": [],
            "STREME hit seqs": [],
            "STREME hit seqs (-)": [],
            "ProfileFinding BED refsites hit": [],
            "STREME BED refsites hit": [],
            "ProfileFinding FIMO refsites hit": [],
            "STREME FIMO refsites hit": [],
            "ProfileFinding MAST refsites hit": [],
            "STREME MAST refsites hit": [],
        } for hitsrc in ['ProfileFinding', 'FIMO', 'MAST']
    }
    for run, wd in run_dirs.items():
        experiment_dirs = [f for f in wd.iterdir() if f.is_dir() and not (f.name.startswith("test") \
                                                                          or f.name.startswith("slurm") \
                                                                          or f.name.startswith("wrong") \
                                                                          or f.name.startswith("_"))]
        result = evaluate_multiple_experiments(experiment_dirs, reevaluate=reevaluate)
        for srceval, srceval_streme, hitsrc in [(result.eval_ProfileFinding, result.eval_STREME, 'ProfileFinding'), 
                                                (result.eval_ProfileFinding_FIMO, result.eval_STREME_FIMO, 'FIMO'), 
                                                (result.eval_ProfileFinding_MAST, result.eval_STREME_MAST, 'MAST')]:
            stats = srceval.getStats(result.sequences, result.negative_sequences, result.refsites)
            stats_streme = srceval_streme.getStats(result.sequences, result.negative_sequences, result.refsites)
            
                # "number_of_hits_test": len(self.hits),
                # "number_of_hits_neg": len(self.negative_hits),
                # "number_of_hits_test_per_seq": len(self.hits)/len(sequences),
                # "number_of_hits_neg_per_seq": len(self.negative_hits)/len(negative_sequences),
                # "test_sequences_with_geq_1_hit": 100*len(self.sequences_with_hits)/len(sequences),
                # "neg_sequences_with_geq_1_hit": 100*len(self.negative_sequences_with_hits)/len(negative_sequences),
                # statsdict[rt] = {
                #     "refsites_hit": 100*len(set(self.hits_on_refsites[rt]))/len(refsites[rt]),
                #     "sequences_with_hits_on_refsites"
            d = comp[hitsrc]
            d["run"].append(run)
            d["sequences"].append(len(result.sequences))
            d["ref. sites (min, max)"].append(f"{min([len(result.refsites[rt]) for rt in result.refsites])}, {max([len(result.refsites[rt]) for rt in result.refsites])}")
            d["ProfileFinding hits"].append(stats['number_of_hits_test'])
            d["ProfileFinding hits (-)"].append(stats['number_of_hits_neg'])
            d["STREME hits"].append(stats_streme['number_of_hits_test'])
            d["STREME hits (-)"].append(stats_streme['number_of_hits_neg'])
            d["ProfileFinding hit seqs"].append(f"{stats['test_sequences_with_geq_1_hit']:.2f} %")
            d["ProfileFinding hit seqs (-)"].append(f"{stats['neg_sequences_with_geq_1_hit']:.2f} %")
            d["STREME hit seqs"].append(f"{stats_streme['test_sequences_with_geq_1_hit']:.2f} %")
            d["STREME hit seqs (-)"].append(f"{stats_streme['neg_sequences_with_geq_1_hit']:.2f} %")
            d["ProfileFinding BED refsites hit"].append(f"{stats['peak_bed.tsv']['refsites_hit']:.2f} %")
            d["STREME BED refsites hit"].append(f"{stats_streme['peak_bed.tsv']['refsites_hit']:.2f} %")
            d["ProfileFinding FIMO refsites hit"].append(f"{stats['peak_fimo.tsv']['refsites_hit']:.2f} %")
            d["STREME FIMO refsites hit"].append(f"{stats_streme['peak_fimo.tsv']['refsites_hit']:.2f} %")
            d["ProfileFinding MAST refsites hit"].append(f"{stats['peak_mast.tsv']['refsites_hit']:.2f} %")
            d["STREME MAST refsites hit"].append(f"{stats_streme['peak_mast.tsv']['refsites_hit']:.2f} %")

    # convert to pandas dataframes
    result = {
        hitsrc: pd.DataFrame(data) for hitsrc, data in comp.items()
    }
        
    return result