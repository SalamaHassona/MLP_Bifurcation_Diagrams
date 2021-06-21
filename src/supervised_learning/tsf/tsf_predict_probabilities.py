import argparse
import pandas as pd
import pickle
import time
from sktime.utils.data_container import detabularize
from joblib import parallel_backend
import torch
import numpy as np


def runge_kutta_4(R, C, x0, y0, z0, data_range, dt, kd, kdmin, L, ts_nth_element=8):
    xp = torch.from_numpy(np.ones((data_range, kd - kdmin), dtype=np.double)).double().cpu()
    xp[:, 0] = x0
    yp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * y0
    zp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * z0
    for k in range(kdmin, kd - 1):
        xx = xp[:, k - kdmin]
        yy = yp
        zz = zp
        kx1 = (yy - xx * pow(zz, -2 / 3)) / L
        ky1 = (R + 1 - yy - R * xx) / (R * C)
        kz1 = xx * xx - zz
        x1 = xx + (dt * kx1) / 2
        y1 = yy + (dt * ky1) / 2
        z1 = zz + (dt * kz1) / 2
        kx2 = (y1 - x1 * pow(z1, -2 / 3)) / L
        ky2 = (R + 1 - y1 - R * x1) / (R * C)
        kz2 = x1 * x1 - z1
        del x1, y1, z1
        x2 = xx + (dt * kx2) / 2
        y2 = yy + (dt * ky2) / 2
        z2 = zz + (dt * kz2) / 2
        kx3 = (y2 - x2 * pow(z2, -2 / 3)) / L
        ky3 = (R + 1 - y2 - R * x2) / (R * C)
        kz3 = x2 * x2 - z2
        del x2, y2, z2
        x3 = xx + (dt * kx3)
        y3 = yy + (dt * ky3)
        z3 = zz + (dt * kz3)
        kx4 = (y3 - x3 * pow(z3, -2 / 3)) / L
        ky4 = (R + 1 - y3 - R * x3) / (R * C)
        kz4 = x3 * x3 - z3
        del x3, y3, z3
        xp[:, k - kdmin + 1] = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * dt
        yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * dt
        zp = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * dt
        del kx1, kx2, kx3, ky1, ky2, ky3, kz1, kz2, kz3
    del yp, zp
    xp = xp.numpy()[:, ::ts_nth_element]
    return xp


def main():
    L, dt, kdmin, kd = args.L, args.dt, int(args.tmin / args.dt), int(args.tmax / args.dt) + 1

    R1, R2 = 6.00, 25.00
    C1, C2 = 2.80, 3.40
    x0, y0, z0 = 0.50, 4.00, 1.00
    NC, NL, NR = 600, 600, 600

    deltaC = (C2 - C1) / (NC - 1)
    deltaR = (R2 - R1) / (NR - 1)

    xyz_points_untrimmed = np.loadtxt(args.data_util_file, delimiter=',')
    xyz_points = np.array([[item[0], item[1], item[2]] for item in xyz_points_untrimmed], dtype=np.double)
    original_untrimmed = np.loadtxt(args.labeled_data_file, delimiter=',')
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < args.threshold else 1]
                              for item in original_untrimmed],
                             dtype=np.double)
    cr_point_list = np.concatenate((cr_point_list, xyz_points), axis=1)

    training_len = 18000

    with open('{model_file_name}.pickle'.format(model_file_name = args.model_file_name), 'rb') as model_pickle:
        model = pickle.load(model_pickle)

    for i in range(0, 10):
        R = torch.from_numpy(cr_point_list[training_len * i:training_len * (i + 1), 0]).double().cpu()
        C = torch.from_numpy(cr_point_list[training_len * i:training_len * (i + 1), 1]).double().cpu()
        x0 = torch.from_numpy(cr_point_list[training_len * i:training_len * (i + 1), 3]).double().cpu()
        y0 = torch.from_numpy(cr_point_list[training_len * i:training_len * (i + 1), 4]).double().cpu()
        z0 = torch.from_numpy(cr_point_list[training_len * i:training_len * (i + 1), 5]).double().cpu()
        x = runge_kutta_4(R, C, x0, y0, z0, training_len, dt, kd, kdmin, L, args.ts_nth_element)
        x = detabularize(pd.DataFrame(x))
        with parallel_backend('threading', n_jobs=args.n_jobs):
            x = model.predict_proba(x)
        with open('{out_file_name}_{i}.pickle'.format(out_file_name=args.out_file_name, i=i), 'wb') \
                as model_probabilities:
            pickle.dump(x, model_probabilities, protocol=pickle.HIGHEST_PROTOCOL)
        del R, C, x0, y0, z0, x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--tmin", type=int, default=1800)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--ts_nth_element", type=int, default=8)
    p.add_argument("--threshold", type=int, default=23)
    p.add_argument("--n_jobs", type=int, default=20)
    p.add_argument("--labeled_data_file", type=str, default="labeled_data_file.txt")
    p.add_argument("--data_util_file", type=str, default="data_util_file.txt")
    p.add_argument("--model_file_name", type=str, default="model_file_name")
    p.add_argument("--out_file_name", type=str, default="out_file_name")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))