from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
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


def test2(path):
    new_db = connect(path)
    sum_rows = 0
    # Iterate over the rows in the original database and filter
    for row in new_db.select():
        # if 'F' in row.formula:
        if row.natoms > 16:
            sum_rows += 1
    return sum_rows


def create_new_db(original_path, new_path, start):
    original_db = connect(original_path)
    new_db = connect(new_path)
    rows = original_db.select()
    total = 0
    for row in rows:
        if row.natoms <= 18:
            if total > start:
                new_db.write(row)
        total += 1
        print(total)


def test3(original_path, new_path):
    iteration = 1
    id_list = []
    original_db = connect(original_path)
    new_db = connect(new_path)
    rows = original_db.select()
    for row in rows:
        if row.natoms <= 18:
            id_list.append(iteration)
            iteration += 1
    new_db.delete(id_list)
    print(len(id_list))


def num_atoms_freq(db):
    freq = {}
    for row in db.select():
        num_atoms = row.natoms
        if num_atoms in freq:
            freq[num_atoms] += 1
        else:
            freq[num_atoms] = 1
    return freq


def element_freq(db):
    freq = {}
    for row in db.select():
        formula = row.formula
        for element in formula:
            if element.isalpha():
                if element in freq:
                    freq[element] += 1
                else:
                    freq[element] = 1
    return freq


def plot_atoms():
    db = connect('qm9.db')

    # freq = element_freq(db)
    freq = num_atoms_freq(db)

    elements = list(freq.keys())
    frequencies = list(freq.values())

    db_len = len(db)
    normalized_frequencies = [element/db_len for element in frequencies]

    sorted_indices = np.argsort(normalized_frequencies)[::-1]
    elements = [elements[i] for i in sorted_indices]
    normalized_frequencies = [normalized_frequencies[i] for i in sorted_indices]

    # Plot the frequencies as a bar plot with custom styling
    plt.bar(elements, normalized_frequencies, color='#e58888', edgecolor='black')
    plt.xlabel('N_atoms', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Number of atoms', fontweight='bold', fontsize=14)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    plt.show()


if __name__ == '__main__':

    # ase db qm9.db -w
    create_new_db('qm9.db', 'test.db', 31310)
    # plot_atoms()
    # print(test2('qm9.db'))
    # new_db = connect('plus_16_atom.db')
    # print(len(new_db))

