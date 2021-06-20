import numpy as np
import time
import argparse
import pickle
from sklearn.cluster import MeanShift


def standardization(x):
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    x = x-x_mean[..., np.newaxis]
    x = x/ x_std[..., np.newaxis]
    return x


def main():
    with open('{residual_x_mean}.pickle'.format(residual_x_mean = args.residual_x_mean), 'rb') \
            as residual_x_mean_pickle:
        residual_x_mean = pickle.load(residual_x_mean_pickle)
    with open('{residual_x_var}.pickle'.format(residual_x_var = args.residual_x_var), 'rb') \
            as residual_x_variance_pickle:
        residual_x_var = pickle.load(residual_x_variance_pickle)

    residual_x = np.concatenate((np.expand_dims(residual_x_var, axis=1), np.expand_dims(residual_x_mean, axis=1)),
                                axis=1)
    meanShift = MeanShift(bandwidth=args.bandwidth, n_jobs=args.n_jobs).fit(residual_x)

    with open('{mean_shift_labels}.pickle'.format(mean_shift_labels = args.out_labels), 'wb') \
            as mean_shift_labels_pickle:
        pickle.dump(meanShift.labels_, mean_shift_labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{mean_shift_cluster_centers}.pickle'.format(mean_shift_cluster_centers = args.out_cluster_centers), 'wb') \
            as mean_shift_cluster_centers_pickle:
        pickle.dump(meanShift.cluster_centers_, mean_shift_cluster_centers_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bandwidth", type=int, default="bandwidth")
    p.add_argument("--n_jobs", type=int, default="n_jobs")
    p.add_argument("--residual_x_mean", type=str, default="residual_x_mean")
    p.add_argument("--residual_x_var", type=str, default="residual_x_var")
    p.add_argument("--out_labels", type=str, default="mean_shift_labels")
    p.add_argument("--out_cluster_centers", type=str, default="mean_shift_cluster_centers")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))