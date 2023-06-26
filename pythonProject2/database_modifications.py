from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
import wandb
import json

def num_atoms_database():
    for i in range(24, 30):
        def cc(row):
            return row.natoms == i
        path = f'./Databases/{str(i)}_atoms.db'

        create_new_db('qm9.db', path, cc)


def create_new_db(original_path, new_path, condition):
    original_db = connect(original_path)
    new_db = connect(new_path)
    rows = original_db.select()
    total = 0
    for row in rows:
        if condition(row):
            new_db.write(row)
        total += 1
        print(total)

def create_new_sub(original_path, new_path, condition):
    original_db = connect(original_path)
    new_db = connect(new_path)
    rows = original_db.select()
    total = 0
    for row in rows:
        if condition(row):
            break
        new_db.write(row)
        total += 1
        print(total)


def subsampling_database(cuttoff, path):
    def subsampling_cond(row):
        return cuttoff < row.id

    new_db = path
    folder_path = './Databases/nAtoms'

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        item_path = item_path.replace('\\', '/')
        create_new_sub(item_path, new_db, subsampling_cond)


if __name__ == '__main__':

    # ase db qm9.db -w
    # num_atoms_database()
    # plot_atoms()
    # print(test2('qm9.db'))
    # new_db = connect('plus_16_atom.db')
    # print(len(new_db))

    original_db = connect('qm9.db')
    new_db = connect('./Databases/Splited/under_18_atoms.db')
    rows = original_db.select()
    total = 0
    for row in rows:
        if row.natoms < 18:
            new_db.write(row)
        total += 1
        print(total)


