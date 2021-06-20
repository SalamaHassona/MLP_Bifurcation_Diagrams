import numpy as np
import time
import argparse
import pickle
from sklearn.cluster import DBSCAN


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

    residual_x = np.concatenate((np.expand_dims(residual_x_var, axis=1),
                                 np.expand_dims(residual_x_mean, axis=1)),
                                axis=1)
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=args.n_jobs).fit(residual_x)

    with open('{dbscan_labels}.pickle'.format(dbscan_labels = args.out_labels), 'wb') \
            as dbscan_labels_pickle:
        pickle.dump(dbscan.labels_, dbscan_labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{dbscan_core_sample_indices}.pickle'
                      .format(dbscan_core_sample_indices = args.out_core_sample_indices), 'wb') \
            as dbscan_core_sample_indices_pickle:
        pickle.dump(dbscan.core_sample_indices_, dbscan_core_sample_indices_pickle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=int, default="eps")
    p.add_argument("--min_samples", type=int, default="min_samples")
    p.add_argument("--n_jobs", type=int, default="n_jobs")
    p.add_argument("--residual_x_mean", type=str, default="residual_x_mean")
    p.add_argument("--residual_x_var", type=str, default="residual_x_var")
    p.add_argument("--out_labels", type=str, default="dbscan_labels")
    p.add_argument("--out_core_sample_indices", type=str, default="dbscan_core_sample_indices")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))