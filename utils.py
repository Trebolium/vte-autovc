from scipy.signal import medfilt
import numpy as np
import math
import matplotlib.pyplot as plt

def saveContourPlots(array_of_contours, file_path, list_of_strings, num_cols):
    # save loss history to a chart

    num_steps = array_of_contours.shape[0]
    step_array = np.arange(num_steps)

    num_contours = array_of_contours.shape[1]
    num_rows = math.ceil(num_contours/num_cols)
    dims = math.ceil(math.sqrt(num_contours))

    x_label = list_of_strings[0]
    y_label = list_of_strings[1]
    labels = list_of_strings[2:]

    plt.figure()

    for i in range(num_contours):
        plt.subplot(num_rows, num_cols, i+1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.ylim(0,1)
        #plt.yticks(np.arange(0, 1, step=0.2))
        plt.plot(step_array, array_of_contours[:,i], 'r--',label=labels[i])
        plt.legend()

    plt.show()
    plt.savefig(file_path)

#if __name__ == '__main__':
#    hist_array = np.random.rand(16,7)
#    print(hist_array)
#    saveContourPlots(hist_array, './test.png', ['x','y','1','2','3','4','5','6','7'], 2)
