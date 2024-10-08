import os
from pathlib import Path
import pandas as pd

def full_experiment():
    wd = Path("/home/ebelm/genomegraph/runs/20240903_replicate_STREME_results/full")
    datadir = Path("/home/ebelm/genomegraph/data/STREME_benchmark_data/")

    # Load the data
    data = pd.read_csv(datadir / "full_ds_ref-motifs.tsv", sep="\t", names=['file', 'ref'])

    # Create the SLURM script for an array job
    script = f"""#!/bin/bash

#SBATCH --job-name=STREME
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=6443
#SBATCH --partition=pinky
#SBATCH --array=0-{len(data)-1}
#SBATCH --time=0-12:00:00
#SBATCH -o {wd}/STREME_%A_%a.out
#SBATCH -e {wd}/STREME_%A_%a.err

# die if SLURM_ARRAY_TASK_ID is not set
if [ -z $SLURM_ARRAY_TASK_ID ]; then
    echo "SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

# get file basenames and ref motifs from array
basenames=({data['file'].str.cat(sep=' ')})
basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
refmotifs=({data['ref'].str.cat(sep=' ')})
refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

# create working directories
mkdir -p {wd}/${{basename}}
pushd {wd}/${{basename}}

echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
echo ""
echo "PATH: ${{PATH}}" # for some reason, otherwise the perl XML parser is not found???
echo ""

# run STREME
source ~/Software/load_MEME.sh

start=`date +%s`

# test run
streme \\
  --p {datadir}/full_ds_primary/$basename.centered100bp.fasta \\
  --n {datadir}/full_ds_control/$basename.centered100bp.fasta.shuf.fasta \\
  --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

tomtom -oc ./tomtom -m ${{refmotif}} -png {datadir}/jolma2013.meme streme/streme.txt

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

def hybrid_experiment():
    wd_base = Path("/home/ebelm/genomegraph/runs/20240903_replicate_STREME_results/hybrid")
    datadir = Path("/home/ebelm/genomegraph/data/STREME_benchmark_data/")

    # Load the data
    data = pd.read_csv(datadir / "hybrid_ds_ref-motifs.tsv", sep="\t", names=['file', 'ref'])

    for sample in (datadir / "hybrid_ds_primary").iterdir():
        if sample.is_dir():
            i = sample.name
            assert (datadir / "hybrid_ds_control" / i).exists(), f"Control sample {i} not found"
            wd = wd_base / i
            wd.mkdir(exist_ok=True)

            # Create the SLURM script for an array job
            script = f"""#!/bin/bash

#SBATCH --job-name=STREME
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=6443
#SBATCH --partition=batch
#SBATCH --array=0-{len(data)-1}
#SBATCH --time=0-12:00:00
#SBATCH -o {wd}/STREME_%A_%a.out
#SBATCH -e {wd}/STREME_%A_%a.err

# die if SLURM_ARRAY_TASK_ID is not set
if [ -z $SLURM_ARRAY_TASK_ID ]; then
    echo "SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

# get file basenames and ref motifs from array
basenames=({data['file'].str.cat(sep=' ')})
basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
refmotifs=({data['ref'].str.cat(sep=' ')})
refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

# create working directories
mkdir -p {wd}/${{basename}}
pushd {wd}/${{basename}}

echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
echo ""
echo "PATH: ${{PATH}}" # for some reason, otherwise the perl XML parser is not found???
echo ""

# run STREME
source ~/Software/load_MEME.sh

start=`date +%s`

# test run
streme \\
  --p {datadir}/hybrid_ds_primary/{i}/$basename.centered100bp.100seq.fasta \\
  --n {datadir}/hybrid_ds_control/{i}/$basename.centered100bp.100seq.shuf.fasta \\
  --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

tomtom -oc ./tomtom -m ${{refmotif}} -png {datadir}/jolma2013.meme streme/streme.txt

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

def diluted_experiment():
    wd_base = Path("/home/ebelm/genomegraph/runs/20240903_replicate_STREME_results/diluted")
    datadir = Path("/home/ebelm/genomegraph/data/STREME_benchmark_data/")

    # Load the data
    data = pd.read_csv(datadir / "full_ds_ref-motifs.tsv", sep="\t", names=['file', 'ref'])

    for sample in (datadir / "diluted_ds_primary").iterdir():
        if sample.is_dir():
            i = sample.name
            assert (datadir / "diluted_ds_control" / i).exists(), f"Control sample {i} not found"
            wd = wd_base / i
            wd.mkdir(exist_ok=True)

            # Create the SLURM script for an array job
            script = f"""#!/bin/bash

#SBATCH --job-name=STREME
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=6443
#SBATCH --partition=batch
#SBATCH --array=0-{len(data)-1}
#SBATCH --time=0-12:00:00
#SBATCH -o {wd}/STREME_%A_%a.out
#SBATCH -e {wd}/STREME_%A_%a.err

# die if SLURM_ARRAY_TASK_ID is not set
if [ -z $SLURM_ARRAY_TASK_ID ]; then
    echo "SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

# get file basenames and ref motifs from array
basenames=({data['file'].str.cat(sep=' ')})
basename=${{basenames[$SLURM_ARRAY_TASK_ID]}}
refmotifs=({data['ref'].str.cat(sep=' ')})
refmotif=${{refmotifs[$SLURM_ARRAY_TASK_ID]}}

# create working directories
mkdir -p {wd}/${{basename}}
pushd {wd}/${{basename}}

echo "Running STREME on $basename with ref motif $refmotif in $(pwd)"
echo ""
echo "PATH: ${{PATH}}" # for some reason, otherwise the perl XML parser is not found???
echo ""

# run STREME
source ~/Software/load_MEME.sh

start=`date +%s`

streme \\
  --p {datadir}/diluted_ds_primary/{i}/$basename.centered100bp.{i}pure.fasta \\
  --n {datadir}/diluted_ds_control/{i}/$basename.centered100bp.{i}pure.shuf.fasta \\
  --oc ./streme --order 2 --minw 8 --maxw 12 --nmotifs 5

tomtom -oc ./tomtom -m ${{refmotif}} -png {datadir}/jolma2013.meme streme/streme.txt

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

def main():
    #full_experiment()
    #hybrid_experiment()
    diluted_experiment()

if __name__ == "__main__":
    main()