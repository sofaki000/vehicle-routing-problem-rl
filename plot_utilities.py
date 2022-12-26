from IPython.core.display import Image
from time import gmtime, strftime
import matplotlib.pyplot as plt

def save_plot_with_multiple_functions_in_same_figure(results, labels, file_name, title):
    plt.clf()
    # Plotting both the curves simultaneously
    colors = ['r', 'g', 'b']
    n = 1  # plot every 1  points
    for i in range(len(results)):
        plt.plot(results[i][::n],  color=colors[i], label=labels[i])

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title(title)

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    file_name = strftime(f"{file_name}Date%Y_%m_%d_Time%H_%M_%S", gmtime())
    plt.savefig(f'{file_name}.png')