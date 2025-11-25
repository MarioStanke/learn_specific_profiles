""" Based on 20250408_repeat_STREME_vs_ProfileFinding.py, does essentially the same but uses both the legacy 
implementation (old Z) and the new one that can handle the new Q. This is to compare the performances on the same Q,
which should be identical. However, the tensorflow gradient might differ sometimes so the results are expected to be
different (different learned profiles). However, if they perform equally well, it might be fine to continue using the
new implementation only.
""" 

import argparse
import os
from pathlib import Path
import pandas as pd

def run_experiment(legacy_Z: bool, 
                   wd: Path, primary_data: Path, control_data: Path, ref_motifs: pd.DataFrame, jolma: Path,
                   jobname: str, n: int, mem: int, partition: str, time: str, array: str = None, pf_config: Path = None,
                   run_pf: bool = True, run_streme: bool = True, run_pf_init: bool = True, no_submit: bool = False):
    (wd / 'slurmout').mkdir(parents=True, exist_ok=True)

    parts = {
        'streme': None,
        'pf': None,
        'pf_init': None
    }
    confstr = '' if pf_config is None else f"  --config {pf_config}"
    legacystr = '\\\n  --legacy-Z' if legacy_Z else ''
    for k, add_wd, add_opt in [('pf', '', ''), ('pf_init', '_init', '\\\n  --do-not-train')]:
        parts[k] = f"""# run ProfileFinding{add_wd}

mkdir -p {wd}/${{basename}}/profilefinding{add_wd}
pushd {wd}/${{basename}}

echo "Running ProfileFinding{add_wd} on $basename with ref motif $refmotif in $(pwd)"
echo ""

pushd /home/ebelm/genomegraph/learn_specific_profiles
python3 20241008_runModel.py \\
  --fasta {primary_data}/${{basename}}.fasta \\
  --out {wd}/${{basename}}/profilefinding{add_wd} \\
  --mode DNA \\
  {confstr}{legacystr}{add_opt}

popd

tomtom -oc ./profilefinding{add_wd}/tomtom -m ${{refmotif}} -png {jolma} profilefinding{add_wd}/profiles.meme

# also store this command in a makefile to repeat it later (usually partly fails for some reason)
echo "source ~/Software/load_MEME.sh" > ./make_tomtom_profilefinding{add_wd}.sh
echo "tomtom -oc ./profilefinding{add_wd}/tomtom -m ${{refmotif}} -png {jolma} profilefinding{add_wd}/profiles.meme" >> ./make_tomtom_profilefinding{add_wd}.sh

popd

"""

    parts['streme'] = f"""# run STREME

mkdir -p {wd}/${{basename}}/streme
pushd {wd}/${{basename}}

echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
echo ""

start=`date +%s`

# test run
streme \\
  --p {primary_data}/$basename.fasta \\
  --n {control_data}/$basename.shuf.fasta \\
  --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

tomtom -oc ./streme/tomtom -m ${{refmotif}} -png {jolma} streme/streme.txt

# also store this command in a makefile to repeat it later (usually partly fails for some reason)
echo "source ~/Software/load_MEME.sh" > ./make_tomtom_streme.sh
echo "tomtom -oc ./streme/tomtom -m ${{refmotif}} -png {jolma} streme/streme.txt" >> ./make_tomtom_streme.sh

popd

"""

    # Create the SLURM script for an array job
    arrayopt = f"{array}" if array else f"0-{len(ref_motifs.index)-1}" 
    script = f"""#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH -N 1
#SBATCH -n {n}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --array={arrayopt}
#SBATCH --time={time}
#SBATCH -o {wd}/slurmout/%A_%a.out
#SBATCH -e {wd}/slurmout/%A_%a.err

# die if SLURM_ARRAY_TASK_ID is not set
if [ -z $SLURM_ARRAY_TASK_ID ]; then
    echo "SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

# get file basenames and ref motifs from array
basenames=({ref_motifs['file'].str.cat(sep=' ')})
basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
refmotifs=({ref_motifs['ref'].str.cat(sep=' ')})
refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

source ~/Software/load_MEME.sh
source ~/genomegraph/.venv/bin/activate

"""
    
    # Add the parts to the script
    if run_pf_init:
        script += parts['pf_init']
    if run_pf:
        script += parts['pf']
    if run_streme:
        script += parts['streme']

    # Write the script to a file
    with open(wd / "run_STREME.sh", "w") as f:
        f.write(script)

    # Submit the job
    if not no_submit:
        os.system(f"sbatch {wd / 'run_STREME.sh'}")



# ----------------------------------------------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(description='Run model vs STREME on STREME benchmark data')
    parser.add_argument('--wd', help = 'Working directory for the run', required = True, type = str)
    parser.add_argument("--config", metavar="PATH", type=str, required=False, 
                        help="JSON object with training configuration. Allowed keys: all arguments to " \
                        + "20241008_runModel.py except `fasta`, `legacy-Z` and `out`. " \
                        + "Keys may not start with dashes (`--`), otherwise there is no distinction between dashes " \
                        + "and underscores (`_`) and they are converted as needed.")
    parser.add_argument('--no-submit', help = 'Do not submit the job, only write the script', action='store_true', 
                        default = False)
    parser.add_argument('--mem', help = 'Memory to allocate for the job', required = False, type = int, 
                        default = 189000)
    parser.add_argument('--partition', help = 'Partition to use for the job', required = False, type = str,
                        default = 'snowball')
    parser.add_argument('--n', help = 'Number of threads to use for the job', required = False, type = int, 
                        default = 72)
    parser.add_argument('--time', help = 'Time to allocate for the job, as string accepted by `#SBATCH --time=`', 
                        required = False, type = str, default = '3-00:00:00')
    parser.add_argument('--array', help = 'Provide a SLURM --array argument to overwrite the default behaviour of' \
                        + 'creating an array with one job per experiment per run.', required=False, type=str, 
                        default=None)
    parser.add_argument('--skip-pf-init', help = 'Skip the ProfileFinding do-not-train run', action='store_true',
                        default = False)
    parser.add_argument('--skip-pf', help = 'Skip the ProfileFinding run', action='store_true', default = False)
    parser.add_argument('--skip-streme', help = 'Skip the STREME run', action='store_true', default = False)
    parser.add_argument('--skip-undiluted', help = 'Skip the undiluted run', action='store_true', default = False)
    parser.add_argument('--skip-diluted', help = 'Skip the diluted runs', action='store_true', default = False)
    parser.add_argument('--skip-hybrid', help = 'Skip the hybrid runs', action='store_true', default = False)
    parser.add_argument('--skip-simulated', help = 'Skip the simulated runs', action='store_true', default = False)
    args = parser.parse_args()

    wd = Path(args.wd)
    wd.mkdir(exist_ok=True)

    datadir = Path("/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited")
    # datadir = Path("/home/matthis/PhD/mnt/brain/genomegraph/data/20250408_STREME_benchmark_revisited") # FOR LOCAL TESTING ONLY!
    assert datadir.exists(), f"Data directory {datadir} does not exist"
    assert (datadir / "target_reference_motifs.tsv").exists(), \
        f"Reference motifs file {datadir / 'target_reference_motifs.tsv'} does not exist"
    assert (datadir / "target_reference_motifs_hybrid.tsv").exists(), \
        f"Reference motifs file {datadir / 'target_reference_motifs_hybrid.tsv'} does not exist"
    assert (datadir / "target_reference_motifs_simulated.tsv").exists(), \
        f"Reference motifs file {datadir / 'target_reference_motifs_simulated.tsv'} does not exist"
    assert (datadir / "jolma2013.meme").exists(), f"Jolma motifs file {datadir / 'jolma2013.meme'} does not exist"
    # check principal existence of the data directories
    assert (datadir / "diluted_dataset" / "1.00" / "primary_sequences").exists(), \
        f"Data directory {datadir / 'diluted_dataset' / '1.00' / 'primary_sequences'} does not exist"
    assert (datadir / "diluted_dataset" / "1.00" / "control_sequences").exists(), \
        f"Control data directory {datadir / 'diluted_dataset' / '1.00' / 'control_sequences'} does not exist"
    
    # check if config file exists
    config = Path(args.config) if args.config else None
    if config:
        assert config.exists(), f"Config file '{config}' does not exist"
    # Load the normal reference data
    refs = pd.read_csv(datadir / "target_reference_motifs.tsv", sep="\t", names=['file', 'ref'])
    
    wd_base = wd
    for legacy_Z in [True, False]:
        print(f"{'='*80}\nRunning experiments with legacy_Z={legacy_Z}\n{'='*80}")
        wd = wd_base / ( "legacy_Z" if legacy_Z else "new_Z" )
        wd.mkdir(exist_ok=True)

        if not args.skip_undiluted:
            # do a full run on undiluted data
            print(f"Running STREME vs ProfileFinding on {datadir / 'diluted_dataset' / '1.00'}")
            run_experiment(legacy_Z=legacy_Z,
                           wd=wd / "full",
                           primary_data=datadir / "diluted_dataset" / "1.00" / "primary_sequences",
                           control_data=datadir / "diluted_dataset" / "1.00" / "control_sequences",
                           ref_motifs=refs,
                           jolma=datadir / "jolma2013.meme",
                           jobname="full_SvPF",
                           n=args.n,
                           mem=args.mem,
                           partition=args.partition,
                           time=args.time,
                           array=args.array,
                           pf_config=config,
                           run_pf=not args.skip_pf,
                           run_streme=not args.skip_streme,
                           run_pf_init=not args.skip_pf_init,
                           no_submit=args.no_submit)
        
        if not args.skip_diluted:
            # do sensitivity analysis on diluted data
            for i in (datadir / "diluted_dataset").iterdir():
                if i.is_dir():
                    try:
                        float(i.name)
                    except ValueError:
                        print(f"Diluted experiments: Skipping {i} as it is not a valid dilution level")
                        continue

                    print(f"Running STREME vs ProfileFinding on {i}")
                    run_experiment(legacy_Z=legacy_Z,
                                   wd=wd / "diluted" / i.name,
                                   primary_data=i / "primary_sequences",
                                   control_data=i / "control_sequences",
                                   ref_motifs=refs,
                                   jolma=datadir / "jolma2013.meme",
                                   jobname=f"diluted_SvPF_{i.name}",
                                   n=args.n,
                                   mem=args.mem,
                                   partition=args.partition,
                                   time=args.time,
                                   array=args.array,
                                   pf_config=config,
                                   run_pf=not args.skip_pf,
                                   run_streme=not args.skip_streme,
                                   run_pf_init=not args.skip_pf_init,
                                   no_submit=args.no_submit)
        
        if not args.skip_hybrid:
            assert (datadir / "hybrid_dataset" / "00" / "primary_sequences").exists(), \
                f"Data directory {datadir / 'hybrid_dataset' / '00' / 'primary_sequences'} does not exist"
            assert (datadir / "hybrid_dataset" / "00" / "control_sequences").exists(), \
                f"Control data directory {datadir / 'hybrid_dataset' / '00' / 'control_sequences'} does not exist"
            
            # do a specificity analysis on hybrid data
            refs_hybrid = pd.read_csv(datadir / "target_reference_motifs_hybrid.tsv", sep="\t", names=['file', 'ref'])
            for rep in (datadir / "hybrid_dataset").iterdir():
                if rep.is_dir():
                    try:
                        int(rep.name)
                    except ValueError:
                        print(f"Hybrid experiments: Skipping {rep} as it is not a valid replicate number")
                        continue

                    print(f"Running STREME vs ProfileFinding on {rep}")
                    run_experiment(legacy_Z=legacy_Z,
                                   wd=wd / "hybrid" / rep.name,
                                   primary_data=rep / "primary_sequences",
                                   control_data=rep / "control_sequences",
                                   ref_motifs=refs_hybrid,
                                   jolma=datadir / "jolma2013.meme",
                                   jobname=f"hybrid_SvPF_{rep.name}",
                                   n=args.n,
                                   mem=args.mem,
                                   partition=args.partition,
                                   time=args.time,
                                   array=args.array,
                                   pf_config=config,
                                   run_pf=not args.skip_pf,
                                   run_streme=not args.skip_streme,
                                   run_pf_init=not args.skip_pf_init,
                                   no_submit=args.no_submit)

        if not args.skip_simulated:
            assert (datadir / "simulated_dataset" / "order_0" / "wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak" / \
                    "0.00" / "primary_sequences").exists(), \
                f"Data directory {datadir / 'simulated_dataset' / 'order_0' / 'wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak' / '0.00' / 'primary_sequences'} does not exist"
            assert (datadir / "simulated_dataset" / "order_0" / "wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak" / \
                    "0.00" / "control_sequences").exists(), \
                f"Control data directory {datadir / 'simulated_dataset' / 'order_0' / 'wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak' /'0.00' / 'control_sequences'} does not exist"
            
            # do an analysis on simulated data
            refs_simulated = pd.read_csv(datadir / "target_reference_motifs_simulated.tsv", 
                                         sep="\t", names=['file', 'ref'])
            for mfreq in \
                (datadir / "simulated_dataset" / "order_0" / "wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak").iterdir():
                if mfreq.is_dir():
                    try:
                        float(mfreq.name)
                    except ValueError:
                        print(f"Simulated experiments: Skipping {mfreq} as it is not a valid motif frequency")
                        continue

                    print(f"Running STREME vs ProfileFinding on {mfreq}")
                    run_experiment(legacy_Z=legacy_Z,
                                   wd=wd / "simulated" / "order_0" / "wgEncodeAwgTfbsSydhK562Gata1UcdUniPk.narrowPeak" / mfreq.name,
                                   primary_data=mfreq / "primary_sequences",
                                   control_data=mfreq / "control_sequences",
                                   ref_motifs=refs_simulated,
                                   jolma=datadir / "jolma2013.meme",
                                   jobname=f"simulated_SvPF_{mfreq.name}",
                                   n=args.n,
                                   mem=args.mem,
                                   partition=args.partition,
                                   time=args.time,
                                   array=args.array,
                                   pf_config=config,
                                   run_pf=not args.skip_pf,
                                   run_streme=not args.skip_streme,
                                   run_pf_init=not args.skip_pf_init,
                                   no_submit=args.no_submit)



if __name__ == "__main__":
    main()