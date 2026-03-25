""" See if the model is compiled with JIT and if not, if we can enable it to make things faster. """

import tensorflow as tf
from pathlib import Path
from modules import training, ProfileFindingSetup, ModelDataSet, SequenceRepresentation

# Enable JIT compilation
tf.config.optimizer.set_jit(enabled="autoclustering")

def main():
    _basepath = None
    for bp in [Path("/home/ebelm/genomegraph/"),
            Path("/home/matthis/PhD/mnt/brain/genomegraph/"),
            Path("/home/ebelm/brain/genomegraph/"),]:
        if bp.exists():
            _basepath = bp
            break
    if _basepath is None:
        raise RuntimeError("Could not find base path.")

    #wd = _basepath / "runs/20251125_compare_implementations_on_STREME_vs_ProfileFinding"
    datadir = _basepath / "data" / "20250408_STREME_benchmark_revisited"
    fasta_file = datadir / "diluted_dataset" / "1.00" / "primary_sequences" / "wgEncodeAwgTfbsBroadK562CtcfUniPk.narrowPeak.fasta"
    sequences = SequenceRepresentation.loadFasta_agnostic(fasta_file)
    if len(sequences) > 1000:
        sequences = sequences[:1000]
    
    genomes = [SequenceRepresentation.Genome([s]) for s in sequences]
    data = ModelDataSet.ModelDataSet(
        genomes,
        ModelDataSet.DataMode.DNA,
        tile_size=100,
        tiles_per_X=1,
        batch_size=1,
        prefetch=3
    )
    setup = ProfileFindingSetup.ProfileFindingTrainingSetup(
        data,
        U=200, k=12, midK=8, s=0, epochs=350, gamma=1.0, l2=0.1, kld=0, mellowmax_alpha=1.0, match_score_factor=0.6, 
        learning_rate=0.1, lr_patience=5, lf_factor=0.75, rho=0, sigma=1, profile_plateau=10, profile_plateau_dev=150,
        n_best_profiles=2, phylo_t=0, Q_order=0, Q_num_models=2, Q_learning_rate=0.01, Q_lr_patience=10,
        Q_lr_factor=0.75, Q_epochs=20
    )
    evaluator = training.MultiTrainingEvaluation()

    training.trainAndEvaluate(
        "JIT_testrun",
        setup,
        evaluator,
        outdir=None,
        rand_seed=42,
    )



if __name__ == "__main__":
    main()