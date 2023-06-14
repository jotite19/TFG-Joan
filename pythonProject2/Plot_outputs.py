import re
import matplotlib.pyplot as plt
import os


def get_Dict_from_file(path):
    result_dict = {}
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Iteration"):
                iteration = line.split(':')[0].strip()
                result = re.search(r'{.*}', line).group()
                if iteration not in result_dict:
                    result_dict[iteration] = []
                result_dict[iteration].append(eval(result))

    val_loss_dict = {}
    for iteration, results in result_dict.items():
        val_loss_dict[iteration] = [result['val_loss'] for result in results]

    return val_loss_dict


def plot_dictionary_data(dictionary, plot_name):

    plt.figure(figsize=(8, 6))
    for it, values in dictionary.items():
        plt.plot(values, label=it, marker='o', linestyle='-', linewidth=1)

    plt.xlabel('Number Atoms', fontsize=12)
    plt.ylabel('Val_loss', fontsize=12)
    plt.xticks(fontsize=10)  # Adjust x-axis tick font size
    plt.yticks(fontsize=10)  # Adjust y-axis tick font size
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


file_path = './Outputs/all_atoms_natoms.txt'
dict = get_Dict_from_file(file_path)
plot_dictionary_data(dict, 'val_loss_plot.png')