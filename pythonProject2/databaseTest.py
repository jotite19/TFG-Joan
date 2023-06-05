from ase.db import connect
import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
def test1(path):
    db = connect(path)
    row = db._get_row(id=1)
    print(row)
    for key in row:
        print('{0:30}: {1}'.format(key, row[key]))

    print()
    data = row.data
    for item in data:
        print('{0:30}: {1}'.format(item, data[item]))

def test3():
    run = wandb.init()
    artifact = run.use_artifact('jotite19/sweep/model-suzgoern:v15', type='model')
    artifact_dir = artifact.download()

def newDb(original_path, new_path):
    # Open the existing ASE database
    original_db = connect(original_path)
    # Create a new ASE database
    new_db = connect('filtered.db')

    new_db._metadata = original_db.metadata
    # Iterate over the rows in the original database and filter
    for row in original_db.select():
        if 'F' in row.formula:
            new_db.write(row)


if __name__ == '__main__':

    #ase db qm9.db -w
    newDb('qm9.db', 'test.db')
