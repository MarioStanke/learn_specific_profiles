import argparse
import os
from pathlib import Path
import pandas as pd

def full_experiment(wd: Path, primary_data: Path, control_data: Path, ref_motifs: pd.DataFrame, jolma: Path,
                    jobname: str, n: int, mem: int, partition: str, time: str):
    (wd / 'slurmout').mkdir(parents=True, exist_ok=True)

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

# create working directories
mkdir -p {wd}/${{basename}}/profilefinding
pushd {wd}/${{basename}}

# run ProfileFinding
echo "Running ProfileFinding on $basename with ref motif $refmotif in $(pwd)"
echo ""

pushd /home/ebelm/genomegraph/learn_specific_profiles

python3 20241008_runModel.py \\
  --fasta {primary_data}/${{basename}}.fasta \\
  --out {wd}/${{basename}}/profilefinding \\
  --mode DNA \\
  --rand-seed 42 \\
  --n-best-profiles 5 \\
  --tiles-per-X 1 --tile-size 100 \\
  --k 12 \\
  --midK 8 \\
  --l2 0.1 \\
  --kld 0.0 \\
  --mellowmax-alpha 1.0

popd

source ~/Software/load_MEME.sh
tomtom -oc ./profilefinding/tomtom -m ${{refmotif}} -png {jolma} profilefinding/profiles.meme

# ---

# run STREME
echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"

start=`date +%s`

# test run
streme \\
  --p {primary_data}/$basename.fasta \\
  --n {control_data}/$basename.shuf.fasta \\
  --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

tomtom -oc ./streme/tomtom -m ${{refmotif}} -png {jolma} streme/streme.txt

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"

popd
"""

    # Write the script to a file
    with open(wd / "run_STREME.sh", "w") as f:
        f.write(script)

    # Submit the job
    os.system(f"sbatch {wd / 'run_STREME.sh'}")

# ----------------------------------------------------------------------------------------------------------------------

# def hybrid_experiment(wd: Path, primary_data: Path, control_data: Path, ref_motifs: pd.DataFrame, 
#                       n: int, mem: int, partition: str):
#     wd_base = Path("/home/ebelm/genomegraph/runs/20240903_replicate_STREME_results/hybrid")
#     datadir = Path("/home/ebelm/genomegraph/data/STREME_benchmark_data/")

#     # Load the data
#     data = pd.read_csv(datadir / "hybrid_ds_ref-motifs.tsv", sep="\t", names=['file', 'ref'])

#     for sample in (datadir / "hybrid_ds_primary").iterdir():
#         if sample.is_dir():
#             i = sample.name
#             assert (datadir / "hybrid_ds_control" / i).exists(), f"Control sample {i} not found"
#             wd = wd_base / i
#             wd.mkdir(exist_ok=True)

#             # Create the SLURM script for an array job
#             script = f"""#!/bin/bash

# #SBATCH --job-name=STREME
# #SBATCH -N 1
# #SBATCH -n 8
# #SBATCH --mem=6443
# #SBATCH --partition=batch
# #SBATCH --array=0-{len(data)-1}
# #SBATCH --time=0-12:00:00
# #SBATCH -o {wd}/STREME_%A_%a.out
# #SBATCH -e {wd}/STREME_%A_%a.err

# # die if SLURM_ARRAY_TASK_ID is not set
# if [ -z $SLURM_ARRAY_TASK_ID ]; then
#     echo "SLURM_ARRAY_TASK_ID is not set"
#     exit 1
# fi

# # get file basenames and ref motifs from array
# basenames=({data['file'].str.cat(sep=' ')})
# basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
# refmotifs=({data['ref'].str.cat(sep=' ')})
# refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

# # create working directories
# mkdir -p {wd}/${{basename}}
# pushd {wd}/${{basename}}

# echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
# echo ""
# echo "PATH: ${{PATH}}" # for some reason, otherwise the perl XML parser is not found???
# echo ""

# # run STREME
# source ~/Software/load_MEME.sh

# start=`date +%s`

# # test run
# streme \\
#   --p {datadir}/hybrid_ds_primary/{i}/$basename.centered100bp.100seq.fasta \\
#   --n {datadir}/hybrid_ds_control/{i}/$basename.centered100bp.100seq.shuf.fasta \\
#   --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

# tomtom -oc ./tomtom -m ${{refmotif}} -png {datadir}/jolma2013.meme streme/streme.txt

# end=`date +%s`
# runtime=$((end-start))
# echo "Runtime: $runtime"

# popd
# """


#             # Write the script to a file
#             with open(wd / "run_STREME.sh", "w") as f:
#                 f.write(script)

#             # Submit the job
#             os.system(f"sbatch {wd / 'run_STREME.sh'}")

# ----------------------------------------------------------------------------------------------------------------------

# def diluted_experiment(wd: Path, primary_data: Path, control_data: Path, ref_motifs: pd.DataFrame,
#                        n: int, mem: int, partition: str):
#     wd_base = Path("/home/ebelm/genomegraph/runs/20240903_replicate_STREME_results/diluted")
#     datadir = Path("/home/ebelm/genomegraph/data/STREME_benchmark_data/")

#     # Load the data
#     data = pd.read_csv(datadir / "full_ds_ref-motifs.tsv", sep="\t", names=['file', 'ref'])

#     for sample in (datadir / "diluted_ds_primary").iterdir():
#         if sample.is_dir():
#             i = sample.name
#             assert (datadir / "diluted_ds_control" / i).exists(), f"Control sample {i} not found"
#             wd = wd_base / i
#             wd.mkdir(exist_ok=True)

#             # Create the SLURM script for an array job
#             script = f"""#!/bin/bash

# #SBATCH --job-name=STREME
# #SBATCH -N 1
# #SBATCH -n 8
# #SBATCH --mem=6443
# #SBATCH --partition=batch
# #SBATCH --array=0-{len(data)-1}
# #SBATCH --time=0-12:00:00
# #SBATCH -o {wd}/STREME_%A_%a.out
# #SBATCH -e {wd}/STREME_%A_%a.err

# # die if SLURM_ARRAY_TASK_ID is not set
# if [ -z $SLURM_ARRAY_TASK_ID ]; then
#     echo "SLURM_ARRAY_TASK_ID is not set"
#     exit 1
# fi

# # get file basenames and ref motifs from array
# basenames=({data['file'].str.cat(sep=' ')})
# basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
# refmotifs=({data['ref'].str.cat(sep=' ')})
# refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

# # create working directories
# mkdir -p {wd}/${{basename}}
# pushd {wd}/${{basename}}

# echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
# echo ""
# echo "PATH: ${{PATH}}" # for some reason, otherwise the perl XML parser is not found???
# echo ""

# # run STREME
# source ~/Software/load_MEME.sh

# start=`date +%s`

# streme \\
#   --p {datadir}/diluted_ds_primary/{i}/$basename.centered100bp.{i}pure.fasta \\
#   --n {datadir}/diluted_ds_control/{i}/$basename.centered100bp.{i}pure.shuf.fasta \\
#   --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

# tomtom -oc ./tomtom -m ${{refmotif}} -png {datadir}/jolma2013.meme streme/streme.txt

# end=`date +%s`
# runtime=$((end-start))
# echo "Runtime: $runtime"

# popd
# """


#             # Write the script to a file
#             with open(wd / "run_STREME.sh", "w") as f:
#                 f.write(script)

#             # Submit the job
#             os.system(f"sbatch {wd / 'run_STREME.sh'}")

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
    parser.add_argument('--mem', help = 'Memory to allocate for the job', required = False, type = int, 
                        default = 189000)
    parser.add_argument('--partition', help = 'Partition to use for the job', required = False, type = str,
                        default = 'snowball')
    parser.add_argument('--n', help = 'Number of threads to use for the job', required = False, type = int, 
                        default = 72)
    parser.add_argument('--time', help = 'Time to allocate for the job, as string accepted by `#SBATCH --time=`', 
                        required = False, type = str, default = '3-00:00:00')
    args = parser.parse_args()

    wd = Path(args.wd)
    wd.mkdir(exist_ok=True)

    datadir = Path("/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited")
    assert datadir.exists(), f"Data directory {datadir} does not exist"
    assert (datadir / "target_reference_motifs.tsv").exists(), \
        f"Reference motifs file {datadir / 'target_reference_motifs.tsv'} does not exist"
    assert (datadir / "jolma2013.meme").exists(), f"Jolma motifs file {datadir / 'jolma2013.meme'} does not exist"
    assert (datadir / "diluted_dataset" / "1.00" / "primary_sequences").exists(), \
        f"Data directory {datadir / 'diluted_dataset' / '1.00' / 'primary_sequences'} does not exist"
    assert (datadir / "diluted_dataset" / "1.00" / "control_sequences").exists(), \
        f"Control data directory {datadir / 'diluted_dataset' / '1.00' / 'control_sequences'} does not exist"

    # Load the reference data
    data = pd.read_csv(datadir / "target_reference_motifs.tsv", sep="\t", names=['file', 'ref'])
    
    full_experiment(wd=wd / "full",
                    primary_data=datadir / "diluted_dataset" / "1.00" / "primary_sequences",
                    control_data=datadir / "diluted_dataset" / "1.00" / "control_sequences",
                    ref_motifs=data,
                    jolma=datadir / "jolma2013.meme",
                    jobname="full_SvPF",
                    n=args.n,
                    mem=args.mem,
                    partition=args.partition,
                    time=args.time)
    #hybrid_experiment()
    #diluted_experiment()

if __name__ == "__main__":
    main()