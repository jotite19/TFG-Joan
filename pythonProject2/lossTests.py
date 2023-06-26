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
    loss = torch.tensor([try_loss for x in range(13, 24)])
    size = torch.tensor([i for i in range(13, 24)]) - 7
    size_factor = (torch.pow(torch.tensor(size), rate) / pow(10, rate + 1))
    factor = size_factor + 1
    result = loss * factor
    return result

if __name__ == '__main__':

    # loss_temps = []
    # loss_list = [1 for i in range(4)]
    # rate_list = [0, 3, 5, 10]
    # loss_list = [1]
    # rate_list = [1]
    '''
    for i in range(len(loss_list)):
        loss_temps.append(get_loss(rate_list[i], loss_list[i]))
    s = torch.tensor([i for i in range(13, 23)])
    plot_loss(s, loss_temps, "A", loss_list)
    '''
    loss_val = 1
    rate = 10
    loss_list = [27.175382614135742, 20.96627426147461, 16.730133056640625, 11.179447174072266, 6.675133228302002,
                 3.131260395050049, 2.0403335094451904, 0.41282033920288086, 0.27136072516441345, 38.7534294128418,
                 39.27702331542969]

    factor_list = get_loss(rate, loss_val)
    print(factor_list)

    for y in range(len(loss_list)):
        loss_list[y] = loss_list[y]/factor_list[y]

    print(loss_list)

