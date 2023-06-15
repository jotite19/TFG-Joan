from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
import wandb
import json


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

    freq = element_freq(db)

    elements = list(freq.keys())
    frequencies = list(freq.values())

    db_len = len(db)
    normalized_frequencies = [element / db_len for element in frequencies]

    sorted_indices = np.argsort(normalized_frequencies)[::-1]
    elements = [elements[i] for i in sorted_indices]
    normalized_frequencies = [normalized_frequencies[i] for i in sorted_indices]

    plt.figure(figsize=(8, 6))
    plt.bar(elements, normalized_frequencies, color='#e58888')
    plt.xlabel('nAtoms', fontsize=12)
    plt.ylabel('Frequencia', fontsize=12)
    plt.title('Frequencia de molecules per nAtoms', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plot_atoms.png')
    plt.show()


def plot_dictionary_data(dictionary, plot_name):
    x = []
    y = []
    for key, value in dictionary.items():
        # Extract the number from the key
        # number = int(key.split('/')[2].split('_')[0])
        number = os.path.splitext(os.path.basename(key))[0]
        x.append(number)
        y.append(value)

    plt.figure(figsize=(8, 6))  # Adjust the figure size
    plt.plot(x, y, marker='o', linestyle='-', color='b', linewidth=2)  # Customize line style and color
    plt.xlabel('Elements', fontsize=12)  # Set x-axis label and font size
    plt.ylabel('Val_loss', fontsize=12)  # Set y-axis label and font size
    plt.title('Loss en validació en funció del Elements', fontsize=14)  # Set plot title and font size
    plt.xticks(fontsize=10)  # Adjust x-axis tick font size
    plt.yticks(fontsize=10)  # Adjust y-axis tick font size
    plt.grid(True)  # Add grid lines
    plt.tight_layout()  # Improve spacing between elements
    plt.savefig(plot_name)


if __name__ == '__main__':

    # ase db qm9.db -w
    # num_atoms_database()
    # plot_atoms()
    # print(test2('qm9.db'))
    new_db = connect('plus_16_atom.db')
    # print(len(new_db))
