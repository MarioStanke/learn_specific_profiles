import argparse
import itertools
import json
import numpy as np
import os
import pathlib

_ABSPATH = pathlib.Path(__file__).parent.resolve() # absolute path of the directory of this file


def check_config(config: dict) -> dict:
    """ Check a (possibly empty) input config dict for valid keys and values, return complete config with defaults for
     missing keys. Values in returned dict are always lists for easy grid creation. """
    allowed_config = {
        'maxexons': (1000, 1, None), # tuples (default value, min value, max value (None if no max))
        'tile_size': (1000, 1, None),
        'tiles_per_X': (7, 1, None),
        'batch_size': (1, 1, None),
        'prefetch': (3, 1, None),
        'n_best_profiles': (2, 1, None),
        'U': (200, 1, None),
        'enforceU': (False, None, None), # if type of tuple[0] is bool, it's a boolean argument
        'minU': (10, 1, None),
        'minOcc': (8, 1, None),
        'overlapTilesize': (6, 0, None),
        'k': (20, 1, None),
        'midK': (12, 1, None),
        's': (6, 0, None),
        'gamma': (0.5, 0, None),
        'l2': (0.01, 0, None),
        'match_score_factor': (0.7, 0, None),
        'learning_rate': (2.0, 0, None),
        'lr_patience': (5, 1, None),
        'lr_factor': (0.5, 0, None),
        'rho': (0, 0, None),
        'sigma': (1, 0, None),
        'phylo_t': (0, 0, None),
        'profile_plateau': (10, 1, None),
        'profile_plateau_dev': (150, 0, None),
    }
    for key in config:
        assert key in allowed_config, f"[check_config] Unknown config key '{key}'"

    full_config = {}
    for key, (default, min_val, max_val) in allowed_config.items():
        if key in config:
            vals = config[key] if type(config[key]) == list else [config[key]]
            for val in vals:
                assert min_val is None or val >= min_val, \
                    f"[check_config] Invalid value for '{key}': {val}, must be >= {min_val}"
                assert max_val is None or val <= max_val, \
                    f"[check_config] Invalid value for '{key}': {val}, must be <= {max_val}"
                
            full_config[key] = vals
        else:
            full_config[key] = [default]

    return full_config




def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--datadir", metavar="PATH", type=str, required=False, help="Data directory",
                        default="/home/ebelm/genomegraph/data/241_species/20231123_subset150_NM_RefSeqBest/20240605_fixed_out_subset150_withEnforced_20_15_20_50_15_20_15_20_mammals/")
    parser.add_argument("--wd", metavar="PATH", type=str, required=True, help="Working directory")
    parser.add_argument("--gridname", metavar="STR", type=str, required=False, help="Optional name of the grid")
    parser.add_argument("--config", metavar="PATH", type=str, help="JSON object with training configuration. Allowed " \
                        + "keys: TODO. Values may be lists, "\
                        + "in which case a grid training over all parameter combinations is performed.")
    parser.add_argument('--mode', help="Data mode, either `DNA` or `Translated`", required=True, type=str, 
                        choices=['DNA', 'Translated'])
    parser.add_argument('--no-softmasking', help = 'Removes softmasking from sequences before training', 
                        required = False, action = 'store_true')
    parser.add_argument('--rand-seed', help = 'Random seed for reproducibility', required = False, type = int)
    parser.add_argument('--isolated-optimization', action='store_true', help='Do not create a grid from all possible ' \
                        + 'configurations, but optimize each parameter individually while setting the remaining ' \
                        + 'parameters to their default values')    
    parser.add_argument("--slurm-partition", metavar="STR", type=str, required=True, help="Partition for slurm jobs")
    parser.add_argument("--slurm-n", metavar="INT", type=int, default=36, help="Number of cores per slurm job")
    parser.add_argument("--slurm-mem", metavar="STR", type=str, default="28G", help="Memory option for slurm jobs")
    parser.add_argument("--slurm-gres", metavar="STR", type=str, help="GPU option for slurm jobs, e.g. `gpu:A100:1`")
    parser.add_argument("--slurm-t", metavar="STR", type=str, default="72:00:00", help="Time option for slurm jobs")
    parser.add_argument("--slurm-max-parallel-jobs", metavar="INT", type=int, help="Set an upper limit on " \
                        + "simulatenously running jobs so other users can still use the cluster")
    parser.add_argument("--dryrun", action="store_true", help="Do not actually submit jobs")
    
    args = parser.parse_args()

    datadir = pathlib.Path(args.datadir)
    assert datadir.exists(), f"Data directory {datadir} does not exist"
    # create working directory
    wd = pathlib.Path(args.wd).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    assert wd.is_dir(), f"Could not create working directory {wd}"
    gridname = f"_{args.gridname}" if args.gridname is not None else "" # name component of working dirs
    # check and parse config, creating grid if necessary
    if args.config is not None:
        configfile = pathlib.Path(args.config)
        assert configfile.is_file, f"Could not find config file {configfile}"
        with open(configfile, 'rt') as fh:
            inputconfig = json.load(fh)
    else:
        inputconfig = {}
    config = check_config(inputconfig)
    
    configs = []
    if args.isolated_optimization:
        # create jobs optimizing each parameter individually
        gridnames = []
        print(f"[DEBUG] >>> Isolated optimization on {config}")
        for key in config.keys():
            if len(config[key]) > 1:
                for val in config[key]:
                    print(f"[DEBUG] >>> Optimizing parameter {key} with value {val}")
                    conf = check_config({key: val}) # use default values except for the current parameter
                    for k in conf.keys():
                        assert len(conf[k]) == 1, f"Expected single value for key {k}, got {conf[k]}"
                        conf[k] = conf[k][0]
                    # duplicates are possible, avoid them
                    if conf not in configs:
                        configs.append(conf)
                        gridnames.append(f"grid{gridname}_{key}_{val}")

        assert len(configs) <= 1000, f"Grid size {len(configs)} is too large, aborting"    
    else:
        # create all-vs-all grid
        cindices = [list(range(len(config[key]))) for key in config.keys()]
        gridsize = 1
        for c in cindices:
            gridsize *= len(c)
        assert gridsize <= 10000, f"Grid size {gridsize} is too large, aborting"

        grid_of_indices = itertools.product(*cindices)
        #print(f"[DEBUG] {cindices=} {list(grid_of_indices)=}")
        for indices in grid_of_indices:
            conf = {key: config[key][i] for key, i in zip(config.keys(), indices)}
            configs.append(conf)

        gridnames = [f"grid{gridname}_{i:05d}" for i in range(len(configs))]

    print(f"Generated {len(configs)} configurations:")
    for i, conf in enumerate(configs):
        print(f"Configuration {i}: {', '.join([f'{key}: {conf[key]}' for key in conf])}")

    # TODO: Change SLURM strategy: use array jobs, i.e. generate a single script that gets an index as SLURM variable
    # and then selects the corresponding configuration from the list of all configurations (needs to be written in the
    # script). Create the sinlge working directories for all configurations and write the corresponding configuration
    # to a file in each directory. Then submit the array job that selects the configuration and wd based on the index.

    # create working directories
    wds = []
    for i, conf in enumerate(configs):
        if conf['phylo_t'] != 0:
            if conf['k'] != 20:
                print(f"Skipping configuration {i}: phylo_t != 0 and k != 20")
                continue
            if args.mode == 'DNA':
                print(f"Skipping configuration {i}: phylo_t != 0 and mode == DNA")
                continue

        if conf['k'] < conf['midK']:
            print(f"Skipping configuration {i}: k < midK")
            continue

        gridwd = wd / gridnames[i]
        gridwd.mkdir(parents=True, exist_ok=True)
        with open(gridwd / "config.json", "w") as f:
            json.dump(conf, f, indent=2)

        assert (gridwd / "config.json").is_file(), f"Could not write config file to {gridwd}"
        wds.append(gridwd)

    # create slurm script
    if args.slurm_gres:
        gpuline1 = f"#SBATCH --gres={args.slurm_gres}"
        gpuline2 = "module load cuda/12.0.0"
    else:
        gpuline1 = ""
        gpuline2 = ""

    moreopts = ""
    if args.no_softmasking:
        moreopts += " --no-softmasking"
    if args.rand_seed is not None:
        moreopts += f" --rand-seed {args.rand_seed}"

    script = f"""#!/bin/bash
#SBATCH -J profile_grid
#SBATCH -N 1
#SBATCH --partition={args.slurm_partition}
#SBATCH -n {args.slurm_n}
#SBATCH --mem={args.slurm_mem}
{gpuline1}
#SBATCH -o {wd}/%j_%a.%N.out
#SBATCH -e {wd}/%j_%a.%N.err

# die if SLURM_ARRAY_TASK_ID is not set
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID is empty, exiting"
    exit 1
fi

echo "On which nodes it executes:"
echo $SLURM_JOB_NODELIST
echo " "
echo "jobname: $SLURM_JOB_NAME"

{gpuline2}

starttime=$(date)
echo "Starttime: "$starttime

wds=({' '.join([str(d) for d in wds])})
gridwd=${{wds[$SLURM_ARRAY_TASK_ID]}}

echo "Running job in $gridwd"

python3 {_ABSPATH / "20240314_runModelVsSTREME_ExonData.py"} \\
  --datadir "{datadir}" \\
  --out "$gridwd" \\
  --mode {args.mode} \\
  --config "$gridwd/config.json" \\
  {moreopts}

echo ""
echo 'Starttime: '$starttime
echo 'Endtime: '$(date)
echo ""
"""
    with open(wd / f"run_grid{gridname}.sh", "w") as f:
        f.write(script)

    print(f"Generated slurm script {wd}/run_grid{gridname}.sh")

    # submit jobs
    joblimit = "" # no limit by default
    if args.slurm_max_parallel_jobs:
        maxjobs = args.slurm_max_parallel_jobs
        if len(wds) > maxjobs:
            joblimit = f"%{maxjobs}"
            print(f"Limiting number of parallel jobs to {maxjobs}")

    cmd = f"sbatch --array=0-{len(wds)-1}{joblimit} {wd}/run_grid{gridname}.sh"
    if not args.dryrun:
        print(f"Running '{cmd}'")
        os.system(cmd)
    else:
        print(f"Skipping job submission (dry run, would be '{cmd}')")


if __name__ == "__main__":
    main()