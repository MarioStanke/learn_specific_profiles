""" Useful functions for evaluating experiments. """

from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from . import utils, plotting


@dataclass
class SingleModelResult:
    experiment: list[str]
    ppval: list[float]

@dataclass
class SingleRunResult:
    streme: SingleModelResult
    profilefinding: SingleModelResult
    profilefinding_init: SingleModelResult



def load_single_run(wd: Path, require_profileFinding: bool = False, require_streme: bool = False, 
                    require_profileFinding_init: bool = False, allow_failed: bool = True) -> SingleRunResult:
    """ Enter the working directory of a single run containing the (40) experiment dirs and load the results """
    assert wd.is_dir(), f"Working directory {wd} does not exist"

    result = SingleRunResult(
        streme=SingleModelResult(experiment=[], ppval=[]),
        profilefinding=SingleModelResult(experiment=[], ppval=[]),
        profilefinding_init=SingleModelResult(experiment=[], ppval=[]),
    )    

    def _load_tomtom(f) -> float:
        result = pd.read_csv(f, sep="\t", comment="#", header=0)
        if len(result) == 0:
            print(f"[WARNING] >>> Empty tomtom result file {f}, returning -1 as ppval!")
            return -1
        
        pval = result["p-value"].min()
        # negative log10 of p-value
        ppval = -math.log10(pval)
        return ppval

    for expdir in wd.iterdir():
        if expdir.is_dir() and expdir.name.startswith("wgEncodeAwgTfbs"):
            for require, model, resultmember in zip(
                [require_profileFinding, require_streme, require_profileFinding_init],
                ['profilefinding', 'streme', 'profilefinding_init'],
                [result.profilefinding, result.streme, result.profilefinding_init]
            ):
                # check if required directories/files exist and die if necessary
                if require:
                    assert (expdir / model).is_dir(), f"Required directory {expdir / model} does not exist"
                else:
                    if not (expdir / model).is_dir():
                        # print(f"[DEBUG] >>> Optional directory {expdir / model} does not exist, skipping")
                        continue

                if not allow_failed:
                    assert (expdir / model / "tomtom").is_dir(), \
                        f"Required directory {expdir / model / 'tomtom'} does not exist"
                    assert (expdir / model / "tomtom" / "tomtom.tsv").is_file(), \
                        f"Required results file {expdir / model / 'tomtom' / 'tomtom.tsv'} does not exist"
                else:
                    if not (expdir / model / "tomtom").is_dir():
                        print(f"[WARNING] >>> Required directory {expdir / model / 'tomtom'} does not exist, skipping")
                        continue
                    if not (expdir / model / "tomtom" / "tomtom.tsv").is_file():
                        print(f"[WARNING] >>> Required results file {expdir / model / 'tomtom' / 'tomtom.tsv'} " \
                                + "does not exist, skipping")
                        continue
            
                # load tomtom results
                tomtom = (expdir / model / "tomtom" / "tomtom.tsv")
                tt_result = _load_tomtom(tomtom)
            
                resultmember.experiment.append(expdir.name)
                resultmember.ppval.append(tt_result)
            
    return result



@dataclass
class TrainedMotif:
    motif: np.ndarray
    shift: int
    ppval: float
    motifname: str



def load_trained_motif(wd: Path, load_idx: int = 0) -> TrainedMotif:
    """ Load the (best) trained motif from a single experiment directory (either profilefinding or streme) together with
        the tomtom p-value and the shift needed to align the motif to the reference.
        
        `load_idx` can be used to load not the best motif (0), but the n-th best motif (1, 2, ...). If the index is out
         of bounds, the last motif is returned. Use -1 to load the worst motif. """
    assert wd.is_dir(), f"Working directory {wd} does not exist"
    assert (wd / "tomtom" / "tomtom.tsv").is_file(), \
        f"Required results file {wd / 'tomtom' / 'tomtom.tsv'} does not exist"
    motiffilename = "profiles.meme" if wd.name in ["profilefinding", "profilefinding_init"] else "streme.txt"
    assert (wd / motiffilename).is_file(), f"Required results file {wd / motiffilename} does not exist"

    tomtom = pd.read_csv(wd / "tomtom" / "tomtom.tsv", sep="\t", comment="#", header=0)
    if len(tomtom) > 0:
        pvals = sorted(tomtom["p-value"].values)
        if load_idx < 0 and abs(load_idx) > len(pvals):
            pval = pvals[0]
            print(f"[WARNING] >>> [{wd.name}] load_idx {load_idx} is negative but out of bounds, " \
                  + "loading best motif instead")
        elif load_idx >= len(pvals):
            pval = pvals[-1]
            print(f"[WARNING] >>> [{wd.name}] load_idx {load_idx} is out of bounds, loading worst motif instead")
        else:
            pval = pvals[load_idx]
        # filter tomtom results to the selected p-value
        tomtom = tomtom[tomtom["p-value"] == pval]
        pval = tomtom["p-value"].values[0]
        shift = -tomtom["Optimal_offset"].values[0]
        ppval = -math.log10(pval)
        motifname = tomtom["Target_ID"].values[0]
        # print(f"[DEBUG] >>> {(wd / motiffilename)}")
        motif: np.ndarray = utils.readMemeFile(wd / motiffilename)[motifname]
        strand = tomtom["Orientation"].values[0]
        assert strand in ["+", "-"], f"Unexpected strand {strand} in tomtom result"
        if strand == "-":
            motif = utils.getMotifRC(motif)
            motifname = f"{motifname} [RC]"
            
    else:
        print(f"[WARNING] >>> Empty tomtom result file {wd / 'tomtom' / 'tomtom.tsv'}, " \
              + "returning first motiv and -1 as ppval!")
        ppval = -1
        motifs = utils.readMemeFile(wd / motiffilename)
        motif: np.ndarray = list(motifs.values())[0]
        motifname = list(motifs.keys())[0]
        shift = 0

    return TrainedMotif(motif=motif, shift=shift, ppval=ppval, motifname=motifname)



def compare_motifs(motifs: list[TrainedMotif], figfile: Path, alphabet: list[str] = [c for c in "ACGT"], 
                   show: bool = False, overwrite: bool = False):
    """ Compare the given trained motifs by plotting them together. """
    if figfile.is_file() and not overwrite:
        print(f"[INFO] >>> Figure file {figfile} already exists, not overwriting and also not running at all")
        return

    assert all([m.motif.shape[1] == len(alphabet) for m in motifs]), f"Motif alphabets do not match {alphabet}"
    shifts = [m.shift for m in motifs]
    shifts = [-min(shifts) + s for s in shifts]  # make all shifts non-negative
    kmax = max([m.motif.shape[0] + shifts[i] for i, m in enumerate(motifs)])
    aligned_motifs = np.zeros((kmax, len(alphabet), len(motifs)), dtype=np.float32)
    for i, m in enumerate(motifs):
        shift = shifts[i]
        aligned_motifs[shift:shift + m.motif.shape[0], :, i] = m.motif

    fig, ax = plt.subplots(len(motifs), 1, figsize=(10, 5 * len(motifs)))
    plotting.plotLogo(aligned_motifs, alphabet, pNames=[m.motifname for m in motifs], ax=ax)
    fig.savefig(figfile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()



def success_rate(ppvals: list[float], threshold: float = 5) -> float:
    """ Calculate the success rate (%) of a model given a SingleModelResult 
    and a ppvalue threshold (usually a number beween 3 and 10) """
    if len(ppvals) == 0:
        return 0
    return 100 * sum([1 for ppval in ppvals if ppval >= threshold]) / len(ppvals)


# === DEPRECATED FUNCTIONS BELOW === #

# @dataclass
# class ExperimentMotifs:
#     streme: TrainedMotif
#     profilefinding: TrainedMotif
#     reference: np.ndarray
#     aligned_motifs: np.ndarray
#     motif_names: list[str]



# def get_experiment_motifs(wd: Path, ref: str, ref_motifs: dict[str, np.ndarray]):
#     """ Load the trained motifs from profilefinding and streme in the given experiment directory and align them
#         to the given reference motif. """
#     assert wd.is_dir(), f"Working directory {wd} does not exist"
#     pfdir = (wd / "profilefinding")
#     stremedir = (wd / "streme")
#     assert pfdir.is_dir() or stremedir.is_dir(), \
#         f"Neither experiment directory {wd / 'profilefinding'} nor {wd / 'streme'} exists"
    
#     ref_motif = ref_motifs[ref]
#     no_trained_motif = TrainedMotif(np.zeros(ref_motif.shape, dtype=ref_motif.dtype), shift=0, ppval=-1, 
#                                     motifname="No motif found") # default if method did not run
#     motif_pf = load_trained_motif(wd / "profilefinding") if pfdir.is_dir() else no_trained_motif
#     motif_streme = load_trained_motif(wd / "streme") if stremedir.is_dir() else no_trained_motif
#     assert set([ref_motif.shape[1], motif_pf.motif.shape[1], motif_streme.motif.shape[1]]) == set([4]), \
#         f"Motif alphabets do not match: {ref_motif.shape}, {motif_pf.motif.shape}, {motif_streme.motif.shape}"
#     # set offsets so that profiles are aligned nicely
#     shift_ref = 0
#     if motif_pf.shift < 0 or motif_streme.shift < 0:
#         shift_ref = abs(min(motif_pf.shift, motif_streme.shift))
#         motif_pf.shift += shift_ref
#         motif_streme.shift += shift_ref

#     kmax = max(ref_motif.shape[0]+shift_ref, 
#                motif_pf.motif.shape[0]+motif_pf.shift, 
#                motif_streme.motif.shape[0]+motif_streme.shift)
#     motifs = np.zeros((kmax, 4, 3), dtype=np.float32)
#     motifs[motif_streme.shift:motif_streme.shift+motif_streme.motif.shape[0], :, 0] = motif_streme.motif
#     motifs[shift_ref:shift_ref+ref_motif.shape[0], :, 1] = ref_motif
#     motifs[motif_pf.shift:motif_pf.shift+motif_pf.motif.shape[0], :, 2] = motif_pf.motif

#     return ExperimentMotifs(streme=motif_streme,
#                             profilefinding=motif_pf,
#                             reference=ref_motif,
#                             aligned_motifs=motifs,
#                             motif_names=[f"STREME {motif_streme.motifname} ({motif_streme.ppval:.2f})", 
#                                          f"Reference {ref}", 
#                                          f"ProfileFinding {motif_pf.motifname} ({motif_pf.ppval:.2f})"])



# def compare_experiment_motifs(experiment_dir: Path, ref_motif: str, motifs: dict[str, np.ndarray], show: bool = False):
#     """ Compare the trained motifs from profilefinding and streme in the given experiment directory to the given
#         reference motif by plotting them together. """
#     assert experiment_dir.is_dir(), f"Experiment directory {experiment_dir} does not exist"
#     figfile = experiment_dir / "compare_motifs.png"
#     try:
#         expmotifs = get_experiment_motifs(experiment_dir, ref_motif, motifs)
#         fig, ax = plt.subplots(3, 1, figsize=(10, 15))
#         plotting.plotLogo(expmotifs.aligned_motifs, [c for c in "ACGT"], pNames=expmotifs.motif_names, ax=ax)
#         if not figfile.is_file():
#             fig.savefig(figfile, bbox_inches="tight")
#         else:
#             print(f"[INFO] >>> Figure file {figfile} already exists, not overwriting")
#         if show:
#             plt.show()
#         plt.close()
#     except AssertionError as e:
#         print(f"[WARNING] >>> {e}")



