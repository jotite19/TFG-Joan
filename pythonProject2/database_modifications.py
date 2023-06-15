from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
import wandb
import json

def num_atoms_database():
    for i in range(25, 26):
        def cc(row):
            return row.natoms == i
        path = f'./Databases/{str(i)}_atoms.db'

        create_new_db('qm9.db', path, 0, cc)


def create_new_db(original_path, new_path, start, condition):
    original_db = connect(original_path)
    new_db = connect(new_path)
    rows = original_db.select()
    total = 0
    for row in rows:
        if condition(row):
            if total > start:
                new_db.write(row)
        total += 1
        print(total)


def custom_condition(row):
    return 'N' in row.formula


if __name__ == '__main__':

    # ase db qm9.db -w
    # num_atoms_database()
    # plot_atoms()
    # print(test2('qm9.db'))
    new_db = connect('plus_16_atom.db')
    # print(len(new_db))
