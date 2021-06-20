import numpy as np
import time
import argparse
import pickle
from seasonal import fit_seasons, adjust_seasons


def standardization(x):
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    x = x-x_mean[..., np.newaxis]
    x = x/ x_std[..., np.newaxis]
    return x


def main():
    Na, Nk = 600, 600

    with open('{training_data}.pickle'.format(training_data = args.training_data), 'rb') as training_data:
        x = pickle.load(training_data)[:,::2]
    x = standardization(x)
    residual_x = np.zeros((Na*Nk, x.shape[1]//2))
    for i in range(0, Na*Nk):
        seasons, trend = fit_seasons(x[i, :])
        if seasons is None:
            residual = x[i, :] - trend
            residual_x[i, :] = residual
        else:
            residual_x[i, :] = adjust_seasons(x[i, :], seasons=seasons) - trend
    with open('{out}.pickle'.format(out=args.out), 'wb') as residual_x_pickle:
        pickle.dump(residual_x, residual_x_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    res_mean = np.mean(residual_x, axis=1)
    with open('{out}_mean.pickle'.format(out=args.out), 'wb') as residual_x_mean_pickle:
        pickle.dump(res_mean, residual_x_mean_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    res_var = np.var(residual_x, ddof=1, axis=1)
    with open('{out}_var.pickle'.format(out=args.out), 'wb') as residual_x_var_pickle:
        pickle.dump(res_var, residual_x_var_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training_data", type=str, default="training_data")
    p.add_argument("--out", type=str, default="residual_x")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))