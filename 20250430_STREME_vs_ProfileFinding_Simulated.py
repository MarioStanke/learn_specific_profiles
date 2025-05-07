""" Basically a copy of the 20241008_STREME_vs_ProfileFinding.py script, but with the following changes:
- only runs the experiments on the simulated data
- uses all simulated data sets (all orders, all modes) """

import argparse
import json
import os
from pathlib import Path
import pandas as pd

def run_experiment(wd: Path, primary_data: Path, control_data: Path, ref_motifs: pd.DataFrame, jolma: Path,
                   jobname: str, n: int, mem: int, partition: str, time: str, pf_config: Path = None,
                   run_pf: bool = True, run_streme: bool = True, run_pf_init: bool = True):
    (wd / 'slurmout').mkdir(parents=True, exist_ok=True)

    parts = {
        'streme': None,
        'pf': None,
        'pf_init': None
    }
    if pf_config is None:
        confstr = ''
        streme_confstr = ''
    else:
        confstr = f"  --config {pf_config}"
        with open(pf_config, 'r') as f:
            config = json.load(f)
        if 'n_best_profiles' in config:
            streme_confstr = f" --nmotifs {config['n_best_profiles']}"
        else:
            streme_confstr = ''

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
  {confstr}{add_opt}

popd

tomtom -oc ./profilefinding{add_wd}/tomtom -m ${{refmotif}} -png {jolma} profilefinding{add_wd}/profiles.meme

# also store this command in a makefile to repeat it later (usually partly fails for some reason)
echo "source ~/Software/load_MEME.sh" > ./make_tomtom_profilefinding{add_wd}.sh
echo "tomtom -oc ./profilefinding{add_wd}/tomtom -m ${{refmotif}} -png {jolma} profilefinding{add_wd}/profiles.meme" >> ./make_tomtom_profilefinding{add_wd}.sh

popd

"""
    # /for

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
  --oc ./streme --order 2 --minw 8 --maxw 12 {streme_confstr}

tomtom -oc ./streme/tomtom -m ${{refmotif}} -png {jolma} streme/streme.txt

# also store this command in a makefile to repeat it later (usually partly fails for some reason)
echo "source ~/Software/load_MEME.sh" > ./make_tomtom_streme.sh
echo "tomtom -oc ./streme/tomtom -m ${{refmotif}} -png {jolma} streme/streme.txt" >> ./make_tomtom_streme.sh

popd

"""

    # Create the SLURM script for an array job
    script = f"""#!/bin/bash

#SBATCH --job-name={jobname}
#SBATCH -N 1
#SBATCH -n {n}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --array=0-{len(ref_motifs.index)-1}
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
    os.system(f"sbatch {wd / 'run_STREME.sh'}")



# ----------------------------------------------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(description='Run model vs STREME on STREME benchmark data')
    parser.add_argument('--wd', help = 'Working directory for the run', required = True, type = str)
    # parser.add_argument('--config', help = 'Path to JSON object with ProfileFinding training configuration. Allowed ' \
    #                     + 'keys are all arguments in parsed form, i.e. no leading dashes and inner dashes (-) must ' \
    #                     + 'be replaced by underscores (_) (e.g. `tile_size` instead of `--tile-size`). Given command ' \
    #                     + 'line arguments overwrite the values in the config file. For arguments neither supplied ' \
    #                     + 'via command line call nor the config file, the default values are used.', 
    #                     required = False, type = str)
    parser.add_argument("--config", metavar="PATH", type=str, required=False, 
                        help="JSON object with training configuration. Allowed keys: all arguments to " \
                        + "20241008_runModel.py except `fasta` and `out`. " \
                        + "Keys may not start with dashes (`--`), otherwise there is no distinction between dashes " \
                        + "and underscores (`_`) and they are converted as needed.")
    parser.add_argument('--mem', help = 'Memory to allocate for the job', required = False, type = int, 
                        default = 189000)
    parser.add_argument('--partition', help = 'Partition to use for the job', required = False, type = str,
                        default = 'snowball')
    parser.add_argument('--n', help = 'Number of threads to use for the job', required = False, type = int, 
                        default = 72)
    parser.add_argument('--time', help = 'Time to allocate for the job, as string accepted by `#SBATCH --time=`', 
                        required = False, type = str, default = '3-00:00:00')
    parser.add_argument('--skip-pf-init', help = 'Skip the ProfileFinding do-not-train run', action='store_true',
                        default = False)
    parser.add_argument('--skip-pf', help = 'Skip the ProfileFinding run', action='store_true', default = False)
    parser.add_argument('--skip-streme', help = 'Skip the STREME run', action='store_true', default = False)
    args = parser.parse_args()

    wd = Path(args.wd)
    wd.mkdir(exist_ok=True)

    datadir = Path("/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited")
    # datadir = Path("/home/matthis/PhD/mnt/brain/genomegraph/data/20250408_STREME_benchmark_revisited") # FOR LOCAL TESTING ONLY!
    assert datadir.exists(), f"Data directory {datadir} does not exist"
    assert (datadir / "jolma2013.meme").exists(), f"Jolma motifs file {datadir / 'jolma2013.meme'} does not exist"
    assert (datadir / "simulated_dataset").exists(), f"Data directory {datadir / 'simulated_dataset'} does not exist"
    order_dirs = list((datadir / "simulated_dataset").glob("order_*"))
    assert len(order_dirs) > 0, f"No order directories found in {datadir}"
    for order_dir in order_dirs:
        mode_dirs = list([d for d in order_dir.glob("*") if d.is_dir()])
        assert len(mode_dirs) > 0, f"No mode directories found in {order_dir}"
        for mode_dir in mode_dirs:
            assert (mode_dir / "target_reference_motifs_simulated.tsv").exists(), \
                f"Reference motifs file {mode_dir / 'target_reference_motifs_simulated.tsv'} does not exist"
            assert (mode_dir / "0.00" / "primary_sequences").exists(), \
                f"Data directory {mode_dir / '0.00' / 'primary_sequences'} does not exist"
            assert (mode_dir / "0.00" / "control_sequences").exists(), \
                f"Control data directory {mode_dir / '0.00' / 'control_sequences'} does not exist"
            # check if config file exists
            config = Path(args.config) if args.config else None
            if config:
                assert config.exists(), f"Config file '{config}' does not exist"
            # do an analysis on simulated data
            refs_simulated = pd.read_csv(mode_dir / "target_reference_motifs_simulated.tsv", 
                                         sep="\t", names=['file', 'ref'])
            for mfreq in mode_dir.iterdir():
                if mfreq.is_dir():
                    try:
                        float(mfreq.name)
                    except ValueError:
                        print(f"Simulated experiments: Skipping {mfreq} as it is not a valid motif frequency")
                        continue

                    assert (mfreq / "primary_sequences").exists(), \
                        f"Data directory {mfreq / 'primary_sequences'} does not exist"
                    assert (mfreq / "control_sequences").exists(), \
                        f"Control data directory {mfreq / 'control_sequences'} does not exist"
                    print(f"Running STREME vs ProfileFinding on {mfreq}")
                    run_experiment(wd=wd / order_dir.name / mode_dir.name / mfreq.name,
                                   primary_data=mfreq / "primary_sequences",
                                   control_data=mfreq / "control_sequences",
                                   ref_motifs=refs_simulated,
                                   jolma=datadir / "jolma2013.meme",
                                   jobname=f"sim_{order_dir.name}_{mode_dir.name}_{mfreq.name}",
                                   n=args.n,
                                   mem=args.mem,
                                   partition=args.partition,
                                   time=args.time,
                                   pf_config=config,
                                   run_pf=not args.skip_pf,
                                   run_streme=not args.skip_streme,
                                   run_pf_init=not args.skip_pf_init)


if __name__ == "__main__":
    main()