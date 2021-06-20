import numpy as np
import time
import argparse
import pickle
from sklearn.cluster import OPTICS


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
    optics = OPTICS(n_jobs=args.n_jobs).fit(residual_x)

    with open('{optics_labels}.pickle'.format(optics_labels = args.out_labels), 'wb') \
            as optics_labels_pickle:
        pickle.dump(optics.labels_, optics_labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{optics_cluster_hierarchy}.pickle'
                      .format(optics_cluster_hierarchy = args.out_cluster_hierarchy), 'wb') \
            as optics_cluster_hierarchy_pickle:
        pickle.dump(optics.cluster_hierarchy_, optics_cluster_hierarchy_pickle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default="n_jobs")
    p.add_argument("--residual_x_mean", type=str, default="residual_x_mean")
    p.add_argument("--residual_x_var", type=str, default="residual_x_var")
    p.add_argument("--out_labels", type=str, default="optics_labels")
    p.add_argument("--out_cluster_hierarchy", type=str, default="optics_cluster_hierarchy")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))