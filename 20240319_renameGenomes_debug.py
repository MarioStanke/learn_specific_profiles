import json
import os
from modules import SequenceRepresentation

traindir = '/home/ebelm/genomegraph/runs/20240314_test_modelVsStreme_slurm_gpu/'
all_genomes = SequenceRepresentation.loadJSONGenomeList(os.path.join(traindir, 'allGenomes.json'))
seGenomes = SequenceRepresentation.loadJSONGenomeList(os.path.join(traindir, '0003_singleExonGenomes.json'))

def rename(genlist: list[list[SequenceRepresentation.Sequence]]):
    for g in genlist:
        for s in g:
            s._regenerateID(recursive=True)

rename(all_genomes)
rename(seGenomes)

with open(os.path.join(traindir, "allGenomes_renamed.json"), "wt") as fh:
    json.dump([g.toDict() for g in all_genomes], fh)

with open(os.path.join(traindir, "0003_singleExonGenomes_renamed.json"), "wt") as fh:
    json.dump([g.toDict() for g in seGenomes], fh)
