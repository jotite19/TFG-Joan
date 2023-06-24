import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    #plt.yscale('log')  # Set y-axis scale to logarithmic
    #plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())  # Display numbers on y-axis
    plt.show()


def get_loss(rate, try_loss):
    my_dict = {3: 0.0014938193225529372, 4: 0.0029876386451058745, 5: 0.003734548306382343, 6: 0.008962915935317624, 7: 0.01568510288680584, 8: 0.0522836762893528, 9: 0.14415356462635845, 10: 0.3936213914926989, 11: 0.858946110467939, 12: 1.7447809687418308, 13: 3.1810882473764797, 14: 5.305299324046756, 15: 7.951600253949284, 16: 10.658400866415207, 17: 12.991746648242893, 18: 13.321880718527094, 19: 13.695335549165327, 20: 9.411808641744782, 21: 9.850991522575345, 22: 3.348396011502409, 23: 4.751839265040894, 24: 0.5325465884901222, 25: 1.4363072786346491, 26: 0.44067670015311644, 27: 0.2658998394144228, 28: 0.2658998394144228, 29: 0.0261418381446764}

    print(try_loss)
    loss = torch.tensor([try_loss for x in range(13, 23)])
    if rate == 0:
        return loss

    size = torch.tensor([i for i in range(13, 23)])  # - 7
    freq = torch.tensor([my_dict[int(key)] for key in size])
    freq = freq.to(loss.device)

    size_factor = torch.pow(torch.tensor(size), rate) / pow(10, rate + 1)

    frequency_factor = freq / 100
    factor = torch.log(size_factor / frequency_factor) - (rate/4.5)
    # factor = size_factor + 1
    result = loss * factor

    return result


loss_temps = []
loss_list = [1 for i in range(4)]
rate_list = [0, 3, 5, 10]
# loss_list = [1]
# rate_list = [1]

for i in range(len(loss_list)):
    loss_temps.append(get_loss(rate_list[i], loss_list[i]))

s = torch.tensor([i for i in range(13, 23)])
plot_loss(s, loss_temps, "A", loss_list)
