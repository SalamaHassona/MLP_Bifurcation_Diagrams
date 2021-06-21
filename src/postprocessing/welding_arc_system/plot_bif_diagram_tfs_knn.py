import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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
    original_untrimmed = np.loadtxt(args.labels_file, delimiter=',')
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < args.threshold else 1] for item in original_untrimmed],
                             dtype=np.double)

    R = cr_point_list[0:180000, 0]
    C = cr_point_list[0:180000, 1]
    data = np.array([])

    for i in range(0, 10):
        with open('{model_file_name}_{i}.pickle'.format(model_file_name=args.model_file_name, i=i), 'rb') as model_pickle:
            data = np.concatenate((data, pickle.load(model_pickle))) if data.size else pickle.load(model_pickle)
    gen_point = np.array([[R[i], C[i], data[i][0], data[i][1]] for i in range(0, 180000)])
    gen_point_list = np.array([[R[i], C[i], 0 if data[i][1] < .5 else 1] for i in range(0, 180000)])

    original = cr_point_list
    net_v2 = gen_point_list

    correct = 0
    true_chaotic = 0
    false_chaotic = 0
    true_periodic = 0
    false_periodic = 0
    new_array = list()
    for i in range(len(original)):
        if original[i][2] == 0 and net_v2[i][2] == 0:
            true_periodic += 1
        if original[i][2] == 1 and net_v2[i][2] == 1:
            true_chaotic += 1
        if original[i][2] == 1 and net_v2[i][2] == 0:
            false_periodic += 1
        if original[i][2] == 0 and net_v2[i][2] == 1:
            false_chaotic += 1
        new_array.append([original[i][0], original[i][1], original[i][2] - net_v2[i][2]])
    print(true_periodic)
    print(true_chaotic)
    print(false_periodic)
    print(false_chaotic)
    accuracy = (true_periodic + true_chaotic) / (true_periodic + true_chaotic + false_periodic + false_chaotic)
    precision = (true_chaotic) / (true_chaotic + false_chaotic)
    recall = (true_chaotic) / (true_chaotic + false_periodic)
    f1score = 2 * precision * recall / (precision + recall)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1score)

    plot_bif_diagram(net_v2, args.save_name)

    new_array = list()
    for i in range(len(original)):
        new_array.append([original[i][0], original[i][1], original[i][2] - net_v2[i, 2]])

    point_array = np.array(new_array)
    plot_bif_diagram(point_array, args.save_name+"_diff")

    y_true = original[:, 2]
    y_probas = gen_point[:, 3]
    yn_probas = gen_point[:, 2]
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_probas)
    lr_auc = roc_auc_score(y_true, y_probas)
    print('Logistic: ROC AUC=%.6f' % (lr_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', label="No Skill")
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic', color="b")
    plt.savefig('{save_name}_roc_auc.png'.format(save_name= args.save_name))
    plt.close()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=int, default=23)
    p.add_argument("--model_file_name", type=str, default="model_file_name.txt")
    p.add_argument("--labels_file", type=str, default="labels_file.txt")
    p.add_argument("--save_name", type=str, default="diagram")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))