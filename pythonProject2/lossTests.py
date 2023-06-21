import torch
import matplotlib.pyplot as plt


def plot_loss(x, y, plot_name, loss_list):
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    for i in range(len(loss_list)):
        plt.plot(x, y[i], label=loss_list[i], marker='o', linestyle='-', linewidth=1)
    plt.xlabel('NAtoms', fontsize=12)  # Set x-axis label and font size
    plt.ylabel('Factor', fontsize=12)  # Set y-axis label and font size
    plt.title('Factor ', fontsize=14)  # Set plot title and font size
    plt.xticks(fontsize=10)  # Adjust x-axis tick font size
    plt.yticks(fontsize=10)  # Adjust y-axis tick font size
    plt.grid(True)  # Add grid lines
    plt.tight_layout()  # Improve spacing between
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.show()


def get_loss(rate, try_loss):
    my_dict = {3: 0.0014938193225529372, 4: 0.0029876386451058745, 5: 0.003734548306382343, 6: 0.008962915935317624, 7: 0.01568510288680584, 8: 0.0522836762893528, 9: 0.14415356462635845, 10: 0.3936213914926989, 11: 0.858946110467939, 12: 1.7447809687418308, 13: 3.1810882473764797, 14: 5.305299324046756, 15: 7.951600253949284, 16: 10.658400866415207, 17: 12.991746648242893, 18: 13.321880718527094, 19: 13.695335549165327, 20: 9.411808641744782, 21: 9.850991522575345, 22: 3.348396011502409, 23: 4.751839265040894, 24: 0.5325465884901222, 25: 1.4363072786346491, 26: 0.44067670015311644, 27: 0.2658998394144228, 29: 0.0261418381446764}

    print(try_loss)
    loss = torch.tensor([try_loss for i in range(10, 28)])
    size = torch.tensor([i for i in range(10, 28)])

    print("Size: ", size)
    freq = torch.tensor([my_dict[int(key)] for key in size])
    freq = freq.to(loss.device)
    print("Freq: ", freq)

    size_factor = (torch.pow(torch.tensor(size), rate) / pow(10, rate + 1))
    print("SizeF: ", size_factor)
    frequency_factor = torch.multiply(freq, loss) * rate
    print("FreqF: ", frequency_factor)

    print("Loss: ", loss)
    temp = loss * (size_factor + frequency_factor)
    print("Loss2", temp)
    return temp


loss_list = [10, 1, 0.1, 0.01, 0.001]
loss_temps = []

for i in loss_list:
    loss_temps.append(get_loss(5, i))

size = torch.tensor([i for i in range(10, 28)])
plot_loss(size, loss_temps, "A", loss_list)
