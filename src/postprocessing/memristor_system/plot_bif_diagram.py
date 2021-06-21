import argparse
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


def plot_bif_diagram(point_array, save_name):
    rows, row_pos = np.unique(point_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(point_array[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(cols), len(rows)), dtype=point_array.dtype)
    pivot_table[col_pos, row_pos] = point_array[:, 2].astype(int)
    x, y = np.meshgrid(rows, cols)
    plt.pcolormesh(x, y, pivot_table)
    plt.colorbar()
    plt.savefig('{save_name}.png'.format(save_name= save_name))
    plt.close()
    plt.show()


def main():
    a1, a2 = 0.2, 0.6
    k1, k2 = 2.3, 2.5
    NK, NA = 600, 600
    deltak = (k2 - k1) / (NK - 1)
    deltaa = (a2 - a1) / (NA - 1)
    k_list = np.array([k1 + deltak * i for i in range(0, 600)])
    a_list = np.array([a1 + deltaa * i for i in range(0, 600)])
    ka_period_list = np.transpose([np.repeat(a_list, len(k_list)), np.tile(k_list, len(a_list))])
    with open('{labels_file}.pickle'.format(labels_file = args.labels_file), 'rb') as labels_pickle:
        labels = pickle.load(labels_pickle)
    predicted_labels = np.concatenate(ka_period_list, np.expand_dims(labels, axis=1), axis=1)
    plot_bif_diagram(predicted_labels, args.save_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_file", type=str, default="labels_file.txt")
    p.add_argument("--save_name", type=str, default="diagram")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))