""" Run grid search on hyperparameters for a model. Try to design it in a reusable way for other models in the future."""

import argparse
from dataclasses import dataclass, field
import itertools
import json
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

@dataclass
class RunOptions:
    run_base_dir: Path
    slurm_job_name: str
    slurm_mem: str
    slurm_n: int
    slurm_partition: str
    slurm_time: str
    slurm_array: str = None
    slurm_gres: str = None
    run_commands: List[str] = field(default_factory=list)

    def copy(self) -> "RunOptions":
        """Create a copy of the RunOptions object."""
        return RunOptions(
            run_base_dir=self.run_base_dir,
            slurm_job_name=self.slurm_job_name,
            slurm_mem=self.slurm_mem,
            slurm_n=self.slurm_n,
            slurm_partition=self.slurm_partition,
            slurm_time=self.slurm_time,
            slurm_array=self.slurm_array,
            slurm_gres=self.slurm_gres,
            run_commands=self.run_commands.copy()
        )
    
    
def start_run(run_options: RunOptions, dry: bool) -> Tuple[Path, Path]:
    """Start a run with the given options and return the paths to the run directory and log file.
    
    Creates the run directory and a slurm script to run the commands. The slurm script is saved in the run_base_dir 
    and is submitted to the slurm scheduler.
    The run directory is created if it does not exist. The run_commands are executed in the run_base_dir.
    """

    assert len(run_options.run_commands) > 0, \
        f"No run commands provided. Please provide at least one command to run. {run_options=}"

    run_dir = run_options.run_base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    slurmout_dir = run_dir / "slurmout"
    slurmout_dir.mkdir(parents=True, exist_ok=True)
    
    slurm_array = f"#SBATCH --array={run_options.slurm_array}\n" if run_options.slurm_array else ""
    slurm_gres = f"#SBATCH --gres={run_options.slurm_gres}\n" if run_options.slurm_gres else ""
    # Create the slurm script
    slurm_script = f"""#!/bin/bash

#SBATCH --job-name={run_options.slurm_job_name}
#SBATCH -N 1
#SBATCH --partition={run_options.slurm_partition}
#SBATCH -n {run_options.slurm_n}
#SBATCH --mem={run_options.slurm_mem}
{slurm_gres}\
{slurm_array}\
#SBATCH --time={run_options.slurm_time}
#SBATCH -o {slurmout_dir}/%A_%a_%x.out
#SBATCH -e {slurmout_dir}/%A_%a_%x.err

pushd {run_dir}

starttime=$(date)
echo "Running job {run_options.slurm_job_name} in $(pwd)"
echo "Starttime: $starttime"
echo ""

"""
    
    slurm_script += "\n".join(run_options.run_commands)
    slurm_script += f"""
popd

endtime=$(date)
echo ""
echo 'Endtime: $endtime'
echo "Job {run_options.slurm_job_name} finished in $(($(date +%s) - $(date -d "$starttime" +%s))) seconds"
echo "That is $(($(($(date +%s) - $(date -d "$starttime" +%s))) / 60)) minutes"
echo "That is $(($(($(date +%s) - $(date -d "$starttime" +%s))) / 3600)) hours"
"""

    # Schedule the job
    with open(run_dir / "slurm_script.sh", "w") as f:
        f.write(slurm_script)
    os.chmod(run_dir / "slurm_script.sh", 0o755)
    if not dry:
        os.system(f"sbatch {run_dir / 'slurm_script.sh'}")
    else:
        print(f"Dry run: sbatch {run_dir / 'slurm_script.sh'}")
    

@dataclass
class ProfileFindingOptions:
    fasta: Path
    out: Path
    mode: str = "DNA"
    config: Path = None
    maxseqs: int = None
    no_softmasking: bool = False
    do_not_train: bool = False
    rand_seed: int = 42
    tile_size: int = 100
    tiles_per_X: int = 1
    batch_size: int = 1
    prefetch: int = 3
    n_best_profiles: int = 5
    U: int = 200
    enforceU: bool = False
    minU: int = 10
    minOcc: int = 8
    overlapTilesize: int = 6
    k: int = 12
    midK: int = 8
    s: int = 0
    gamma: float = 1.0
    l2: float = 0.1
    kld: float = 0.0
    mellowmax_alpha: float = 1.0
    match_score_factor: float = 0.7
    learning_rate: float = 2.0
    lr_patience: int = 5
    lr_factor: float = 0.75
    rho: float = 0.0
    sigma: float = 1.0
    phylo_t: float = 0.0
    profile_plateau: int = 10
    profile_plateau_dev: float = 150

    def create_command(self, pf_dir: Path = Path("/home/ebelm/genomegraph/learn_specific_profiles")) -> str:
        """Create the command to run ProfileFinding."""
        cmd = f"pushd {pf_dir}\n"
        cmd += "python3 20241008_runModel.py \\\n"
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, bool):
                    cmd += f"  --{key.replace('_', '-')} \\\n" if value else ""
                else:
                    cmd += f"  --{key.replace('_', '-')} {value} \\\n"

        cmd += "popd\n"
        return cmd
    

def generate_grid(grid_config: Dict[str, Any], isolated_optimization: bool = False) -> List[Dict[str, Any]]:
    """Generate a grid of all possible combinations of the given parameters.
    
    The grid is generated by creating a list of dictionaries, where each dictionary represents a combination of 
    parameters. The keys of the dictionaries are the parameter names and the values are the parameter values.
    """
    grid = []
    if isolated_optimization:
        opt_keys = [k for k in grid_config.keys() if isinstance(grid_config[k], list)]
        default_config = {k: v[0] if isinstance(v, list) else v for k, v in grid_config.items()}
        grid.append(default_config)
        for key in opt_keys:
            assert len(grid_config[key]) >= 1, f"Parameter {key} has no values in the grid config. " \
                + "Please provide at least one value for this parameter."
            if len(grid_config[key]) > 1:
                # create new grids for this parameter only
                for value in grid_config[key][1:]:
                    config = default_config.copy()
                    config[key] = value
                    grid.append(config)
            elif len(grid_config[key]) == 1:
                print(f"Warning: parameter {key} has only one value in the grid config. " \
                      + "This parameter will not be optimized, only the single (default) value will be used.")

    else:        
        keys = list(grid_config.keys())
        values = [v if isinstance(v, list) else [v] for v in grid_config.values()]
        gridsize = 1
        for value in values:
            gridsize *= len(value)
        print(f"Grid size: {gridsize} ({len(keys)} parameters)")
        assert gridsize < 10000, f"Grid size is too large ({gridsize}). Please reduce the number of parameters " \
                                    + "or the number of values per parameter."
        assert gridsize > 0, f"Grid size is 0. Please provide at least one value for each parameter in the grid config."
        # create a grid for all parameters
        for value_combination in itertools.product(*values):
            grid.append(dict(zip(keys, value_combination)))
    
    return grid


def create_ProfileFindingOptions(single_grid_config: Dict[str, Any], fasta: Path, out: Path) -> ProfileFindingOptions:
    """Create a ProfileFindingOptions object from the given grid config."""
    profile_finding_options = ProfileFindingOptions(
        fasta=fasta,
        out=out
    )
    for k, value in single_grid_config.items():
        assert not k.startswith("-"), \
            f"Parameter {k} starts with a dash. Please remove the leading dashes from the parameter name."
        key = k.replace("-", "_") # convert to python style if necessary
        assert hasattr(profile_finding_options, key), \
            f"Parameter {key} is not a valid parameter for ProfileFinding. Please check the parameter name."
        assert not isinstance(value, list), \
            f"Parameter {key} is a list. Please provide a single value for this parameter."
        # set the value
        setattr(profile_finding_options, key, value)

    return profile_finding_options


def main():
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--wd", metavar="PATH", type=str, required=True, help="Working directory")
    parser.add_argument("--config", metavar="PATH", type=str, required=True, 
                        help="JSON object with training configuration. Allowed keys: all arguments to " \
                        + "20241008_runModel.py except `fasta` and `out`. " \
                        + "Keys may not start with dashes (`--`), otherwise there is no distinction between dashes " \
                        + "and underscores (`_`) and they are converted as needed. "\
                        + "Values may be lists with >= 2 elements, in which case a grid training " \
                        + "over all parameter combinations is performed (unless --isolated-optimization is set). " \
                        + "A value that is not a list is used for every grid run, i.e. overwriting the default value.")
                        # for future reference: in case there is an argument that accepts a value list itself, 
                        #   it's probably easiest to pass these lists as strings in the config JSON
    parser.add_argument("--slurm-partition", metavar="STR", type=str, required=True, help="Partition for slurm jobs")
    parser.add_argument("--datadir", metavar="PATH", type=str, required=False, help="Data directory",
                        default="/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited/diluted_dataset/1.00")
    parser.add_argument("--target-motifs", metavar="PATH", type=str, required=False, help="Target reference motifs",
                        default="/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited/target_reference_motifs.tsv")
    parser.add_argument("--jolma", metavar="PATH", type=str, required=False, help="Jolma database",
                        default="/home/ebelm/genomegraph/data/20250408_STREME_benchmark_revisited/jolma2013.meme")
    parser.add_argument("--gridname", metavar="STR", type=str, required=False, help="Optional name of the grid")
    # parser.add_argument('--rand-seed', help = 'Random seed for reproducibility', required = False, type = int, default=42)
    parser.add_argument('--isolated-optimization', action='store_true', help='Do not create a grid from all possible ' \
                        + 'configurations, but optimize each parameter individually while setting the remaining ' \
                        + 'parameters to their default values. For all config values that are lists, ' \
                        + 'only the first value is used as default when this parameter is not currently optimized!')
    parser.add_argument("--slurm-n", metavar="INT", type=int, default=36, help="Number of cores per slurm job")
    parser.add_argument("--slurm-mem", metavar="STR", type=str, default="28G", help="Memory option for slurm jobs")
    parser.add_argument("--slurm-gres", metavar="STR", type=str, help="GPU option for slurm jobs, e.g. `gpu:A100:1`")
    parser.add_argument("--slurm-time", metavar="STR", type=str, default="72:00:00", help="Time option for slurm jobs")
    parser.add_argument("--slurm-max-parallel-jobs", metavar="INT", type=int, help="Set an upper limit on " \
                        + "simulatenously running jobs so other users can still use the cluster")
    parser.add_argument("--dryrun", action="store_true", help="Do not actually submit jobs")
    args = parser.parse_args()

    wd = Path(args.wd)
    wd.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.datadir)
    assert datadir.is_dir(), f"Data directory {datadir} is not a directory"
    assert (datadir / "primary_sequences").is_dir(), f"Data directory {datadir} does not contain primary sequences"
    assert (datadir / "control_sequences").is_dir(), f"Data directory {datadir} does not contain control sequences"
    target_motifs = Path(args.target_motifs)
    assert target_motifs.is_file(), f"Target reference motifs file {target_motifs} is not a file"
    jolma = Path(args.jolma)
    assert jolma.is_file(), f"Jolma database file {jolma} is not a file"

    ref_motifs = pd.read_csv(target_motifs, sep="\t", names=['file', 'ref'])

    config_file = Path(args.config)
    assert config_file.is_file(), f"Config file {config_file} is not a file"
    with open(config_file, "r") as f:
        grid_config = json.load(f)    
    
    # basic run options
    basic_run_options = RunOptions(
        run_base_dir=wd,
        slurm_job_name="gridsearch",
        slurm_mem=args.slurm_mem,
        slurm_n=args.slurm_n,
        slurm_partition=args.slurm_partition,
        slurm_time=args.slurm_time,
        slurm_array=f"0-{len(ref_motifs) - 1}{f'%{args.slurm_max_parallel_jobs}' if args.slurm_max_parallel_jobs else ''}",
        slurm_gres=args.slurm_gres
    )

    grid = generate_grid(grid_config, args.isolated_optimization)
    
    for i, single_grid_config in enumerate(grid):
        run_options = basic_run_options.copy()
        run_options.slurm_job_name = f"gridsearch_{i:05d}"
        run_options.run_base_dir = wd / f"gridsearch_{i:05d}"

        commands = []
        commands.append(f"""
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
""")
        fasta = datadir / "primary_sequences" / "${basename}.fasta"
        out = run_options.run_base_dir / "${basename}"
        pf_options = create_ProfileFindingOptions(single_grid_config, 
                                                  fasta=fasta, 
                                                  out=out)
        commands.append(pf_options.create_command())
        commands.append(f"""
source ~/Software/load_MEME.sh
tomtom -oc {out / 'tomtom'} -m ${{refmotif}} -png {jolma} {out / 'profiles.meme'}

# also store this command in a makefile to repeat it later (usually partly fails for some reason)
echo "source ~/Software/load_MEME.sh" > ./make_tomtom.sh
echo "tomtom -oc {out / 'tomtom'} -m ${{refmotif}} -png {jolma} {out / 'profiles.meme'}" >> ./make_tomtom.sh
""")
        
        run_options.run_commands = commands

        # start the run
        start_run(run_options, args.dryrun)
        print(f"Started run {i} with config: {single_grid_config}")


if __name__ == "__main__":
    main()