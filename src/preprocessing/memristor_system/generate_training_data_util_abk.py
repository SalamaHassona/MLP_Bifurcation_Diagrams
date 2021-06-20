import numpy as np
import time
import torch
import argparse
import pickle


def generate_data_labels(ka_period_list, x0, y0, z0, b, dt, kd, kdmin):
    a = torch.from_numpy(ka_period_list[:, 0]).double().cpu()
    k = torch.from_numpy(ka_period_list[:, 1]).double().cpu()
    xp = torch.from_numpy(np.ones(360000) * x0).double().cpu()
    xout = torch.from_numpy(np.ones((360000, (kd - kdmin) // 200)) * x0).double().cpu()
    yp = torch.from_numpy(np.ones(360000) * y0).double().cpu()
    zp = torch.from_numpy(np.ones(360000) * z0).double().cpu()

    for i in range(1, kd):
        xx = xp
        yy = yp
        zz = zp
        kx1 = a * torch.log(b * torch.cosh(yy)) - a * torch.log(xx + b)
        ky1 = k * yy - zz - (xx + b) * torch.tanh(yy)
        kz1 = (2 * k + 1) * yy - 2 * zz
        x1 = xx + (dt * kx1) / 2
        y1 = yy + (dt * ky1) / 2
        z1 = zz + (dt * kz1) / 2
        kx2 = a * torch.log(b * torch.cosh(y1)) - a * torch.log(x1 + b)
        ky2 = k * y1 - z1 - (x1 + b) * torch.tanh(y1)
        kz2 = (2 * k + 1) * y1 - 2 * z1
        x2 = xx + (dt * kx2) / 2
        y2 = yy + (dt * ky2) / 2
        z2 = zz + (dt * kz2) / 2
        kx3 = a * torch.log(b * torch.cosh(y2)) - a * torch.log(x2 + b)
        ky3 = k * y2 - z2 - (x2 + b) * torch.tanh(y2)
        kz3 = (2 * k + 1) * y2 - 2 * z2
        x3 = xx + (dt * kx3)
        y3 = yy + (dt * ky3)
        z3 = zz + (dt * kz3)
        kx4 = a * torch.log(b * torch.cosh(y3)) - a * torch.log(x3 + b)
        ky4 = k * y3 - z3 - (x3 + b) * torch.tanh(y3)
        kz4 = (2 * k + 1) * y3 - 2 * z3
        xp = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * dt
        yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * dt
        zp = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * dt
        # xml[k] = xp
        if i >= kdmin and i % 10 == 0:
            xout[:, (i - kdmin) // 200] = xp
    return xout


def main():
    a1, a2 = 0.2, 0.6
    k1, k2 = 2.3, 2.5
    x0, y0, z0 = 0.0, 1.00, 0.00
    Na, Nb, Nk = 600, 600, 600

    deltak = (k2 - k1) / (Na - 1)
    deltaa = (a2 - a1) / (Nk - 1)

    k_list = np.array([k1 + deltak * i for i in range(0, 600)])
    a_list = np.array([a1 + deltaa * i for i in range(0, 600)])
    ka_period_list = np.transpose([np.repeat(a_list, len(k_list)), np.tile(k_list, len(a_list))])
    ka_period_list = np.append(ka_period_list, np.zeros((Na*Nk, 1)), axis=1)

    x = generate_data_labels(ka_period_list, x0, y0, z0, args.b,
                                  args.dt, int(args.tmax/args.dt)+1, int(args.tmin/args.dt)+1)
    with open('{out}.pickle'.format(out = args.out), 'wb') as training_data:
        pickle.dump(x.detach().numpy(), training_data, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=float, default=2.4082 * 10**(-5))
    p.add_argument("--dt", type=float, default=0.0001)
    p.add_argument("--tmin", type=int, default=700)
    p.add_argument("--tmax", type=int, default=1000)
    p.add_argument("--out", type=str, default="training_data")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))