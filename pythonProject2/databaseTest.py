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
    # plot_atoms()

    # Connect to the ase.db file
    db = connect('qm9.db')

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
    desired_elements = ['C', 'F', 'N', 'O']
    df_filtered = df[df['y'].isin(desired_elements)]

    # Create a pivot table to prepare data for heatmap
    pivot_table = df_filtered.pivot_table(index='y', columns='x', values='presence', fill_value=0)

    # Normalize the values in each row of the pivot table to add up to 1
    row_normalized_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(row_normalized_table, annot=True, cmap='YlGnBu')

    # Set the labels for x and y axes
    plt.xlabel('Number of Atoms')
    plt.ylabel('Elements')

    # Set the title of the heatmap
    plt.title('Heatmap of Atoms and Elements')

    # Display the heatmap
    plt.show()
