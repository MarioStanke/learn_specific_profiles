import argparse
from datetime import datetime, timedelta
import itertools
import json
import os
import pandas as pd
import re
import subprocess

parser = argparse.ArgumentParser(description = "Evaluate runtima and memory on a grid of parameters for profile training",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")

#scriptArgs.add_argument("--bool",
#                        dest = "",
#                        action = 'store_true',
#                        help = "")

scriptArgs.add_argument("--out",
                        dest = "out", 
                        metavar = "FILE", 
                        type=str,
                        required = True,
                        help="File to store result in")
scriptArgs.add_argument("--parameters",
                        dest = "parameters", 
                        metavar = "FILE", 
                        type=argparse.FileType("r"),
                        required = True,
                        help="File containing JSON object with training.py script \
                              arguments as keys and lists of parameters as values")
scriptArgs.add_argument("--tmp",
                        dest = "tmp", 
                        metavar = "PATH", 
                        type=str,
                        default=".",
                        help="Directory to store temporary file in")
                        
args = parser.parse_args()

assert os.path.exists(args.tmp), "No file or directory "+str(args.tmp)
assert os.path.isdir(args.tmp), str(args.tmp)+" is no directory"

timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
tmpfile = os.path.join(args.tmp, timestamp+"_tmpfile.txt")
assert not os.path.exists(tmpfile), str(tmpfile)+" already exists"

with open(args.parameters.name, "rt") as fh:
    parameters = json.load(fh)

assert not os.path.exists(args.out), str(args.out)+" already exists"
with open(args.out, "wt") as fh:
    fh.write("") # try writing something to out



# create grid and run training for each combination, evaluation the time and memory needed

vals = []
keys = []
for k in parameters.keys():
    keys.append(k)
    if isinstance(parameters[k], list):
        vals.append(parameters[k])
    else:
        vals.append([parameters[k]])

assert len(vals) == len(keys), str(keys)+"\n\n"+str(vals)

grid = list(itertools.product(*vals))
for g in grid:
    assert len(g) == len(keys), str(g)+"\n\n"+str(keys)

print("[DEBUG] >>> keys:", keys)
print("[DEBUG] >>> grid:\n", grid)

reRAM = re.compile("\s*Maximum resident set size \(kbytes\): (\d+)")
def getTimedeltaStr(time: timedelta):
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    tstr = str(int(hours))+":"+str(int(minutes))+":"+str(seconds)
    return tstr

history = []
for g in grid:
    paramstr = ""
    for i in range(len(keys)):
        paramstr += " --"+str(keys[i])+" "+str(g[i])

    print("[DEBUG] >>> i:", i, "\nparamstr:", paramstr)
    cmd = "/usr/bin/time -v python3 ./training.py"+paramstr+" |& tee "+str(tmpfile)
    print("[DEBUG] >>> Running command ", cmd)
    start = datetime.now()
    returncode = subprocess.run([cmd], shell=True, executable='/bin/bash')
    end = datetime.now()
    time = (end-start).total_seconds()
    runtime = getTimedeltaStr(time)
    print("[DEBUG] >>> runtime:", runtime)
    memory = None
    with open(tmpfile, 'rt') as fh:
        for line in fh:
            m = reRAM.match(line)
            if m:
                memory = float(m.group(1))/1024

    assert memory is not None, "Could not determine memory requirement from run"
    os.remove(tmpfile)

    h = [str(p) for p in g]
    h.append(runtime)
    h.append(memory)
    history.append(h)

c = list(keys)
c.append("runtime")
c.append("memory")

dfHist = pd.DataFrame(history, columns = c)
dfHist.to_csv(args.out, index = False)
print("[DEBUG] >>> history:\n", dfHist)