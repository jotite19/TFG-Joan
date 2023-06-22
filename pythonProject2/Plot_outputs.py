import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def get_dict_from_file(path):
    result_dict = {}
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            iteration = line.split(':')[0].strip()
            result = re.search(r'{.*}', line).group()
            if iteration not in result_dict:
                print(iteration)
                result_dict[iteration] = []
            result_dict[iteration].append(eval(result))

    val_loss_dict = {}
    for rate, results in result_dict.items():
        val_loss_dict[rate] = [result['val_loss'] for result in results]

    return val_loss_dict


def plot_dictionary_data(dictionary, plot_name, type_plot):

    elements = [x for x in range(13, 24)]
    # elements = ['C', 'H', 'O', 'N', 'F']

    plt.figure(figsize=(8, 6))

    for it, values in dictionary.items():
        lab = it
        plt.plot(elements, values, label=lab, marker='o', linestyle='-', linewidth=1)

    plt.xlabel('Elements', fontsize=12)
    plt.ylabel('Val_loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())  # Display numbers on y-axis

    plt.show()


def modify_keys(dictionary):
    modified_dict = {}
    for key, value in dictionary.items():
        if key.startswith('Iteration'):
            rate_key = 'Rate: ' + key.split('Iteration')[1]
            modified_dict[rate_key] = value
        else:
            modified_dict[key] = value
    return modified_dict


if __name__ == '__main__':

    file_path = './Outputs/loss_factor_v1.txt'
    inputs = modify_keys(get_dict_from_file(file_path))
    plot_dictionary_data(inputs, 'val_loss_plot.png', 'A')

