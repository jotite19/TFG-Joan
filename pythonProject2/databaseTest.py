from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
import wandb
import json
import pandas as pd
import seaborn as sns


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


def test2(path):
    new_db = connect(path)
    sum_rows = 0
    # Iterate over the rows in the original database and filter
    for row in new_db.select():
        # if 'F' in row.formula:
        if row.natoms > 16:
            sum_rows += 1
    return sum_rows


if __name__ == '__main__':

    # ase db qm9.db -w
    # num_atoms_database()
    test2()


