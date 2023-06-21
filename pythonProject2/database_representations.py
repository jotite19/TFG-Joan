from ase.db import connect
import matplotlib.pyplot as plt
import schnetpack as spk
import numpy as np
import os
import wandb
import json
import seaborn as sns
import pandas as pd


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


def freq_dict(path):
    db = connect(path)

    freq = num_atoms_freq(db)

    my_dict = freq

    total_sum = sum(my_dict.values())

    # Step 2, 3, and 4: Divide each value by the sum and create a new dictionary
    for key, value in my_dict.items():
        my_dict[key] = (value / total_sum) * 100

    sorted_dict = dict(sorted(my_dict.items()))

    print(sorted_dict)


def plot_atoms(path):
    db = connect(path)

    freq = num_atoms_freq(db)

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
        y.append(value)
    x = [x for x in range(13, 24)]  # Array for natoms
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

def plot_heat_map(path):
    # Connect to the ase.db file
    db = connect(path)

    # Fetch the data from the database
    data = db.select()

    # Create a dictionary to store the count of rows for each (element, natoms) combination
    element_counts = {}

    # Iterate over the rows in the database
    for row in data:
        natoms = row.natoms
        if 10 <= natoms <= 25:
            elements = [atom for atom in row.formula if atom.isalpha()]
            for element in elements:
                key = (element, natoms)
                if key in element_counts:
                    element_counts[key] += 1
                else:
                    element_counts[key] = 1

    # Create lists to store the x and y values
    x_values = []
    y_values = []
    presence_values = []

    # Iterate over the (element, natoms) combinations
    for (element, natoms), count in element_counts.items():
        x_values.append(natoms)
        y_values.append(element)
        presence_values.append(count)

    # Create a DataFrame from the x, y, and presence values
    df = pd.DataFrame({'x': x_values, 'y': y_values, 'presence': presence_values})

    # Filter the DataFrame to include only the desired elements
    desired_elements = ['C']
    df_filtered = df[df['y'].isin(desired_elements)]

    # Create a pivot table to prepare data for heatmap
    pivot_table = df_filtered.pivot_table(index='y', columns='x', values='presence', fill_value=0)

    # Normalize the values in each row of the pivot table to add up to 1
    row_normalized_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 2))
    sns.heatmap(row_normalized_table, annot=True, cmap='YlGnBu')

    # Set the labels for x and y axes
    plt.xlabel('Number of Atoms')
    plt.ylabel('Elements')

    # Set the title of the heatmap
    plt.title('Heatmap of Atoms and Elements')

    # Display the heatmap
    plt.show()


if __name__ == '__main__':

    # ase db qm9.db -w
    # num_atoms_database()
    # plot_atoms()
    # print(test2('qm9.db'))
    # new_db = connect('plus_16_atom.db')
    # print(len(new_db))

    #plot_atoms('6000_subsample.db')
    # plot_heat_map('qm9.db')
    # plot_heat_map('under_sample.db')

    #freq_dict('qm9.db')

    e = 3 % 353
    print(e)
