import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def main():
    #Convert from txt to NDarray
    arr = np.loadtxt("test/test.txt", dtype=int, delimiter=",")

    #Create a heatmap from NDarray
    heat = sn.heatmap(arr)

    #Save figure to file (Optional)
    fig = heat.get_figure()
    fig.savefig('test2.png', dpi = 400)

    #Show figure
    plt.show()

main()