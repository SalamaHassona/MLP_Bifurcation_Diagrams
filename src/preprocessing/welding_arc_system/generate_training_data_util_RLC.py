import numpy as np
import time
import torch
import argparse


def runge_kutta_4(R, C, x0, y0, z0, L, data_range, dt, kdmin):
    yp = torch.from_numpy(np.ones(data_range) * y0).double().cpu()
    zp = torch.from_numpy(np.ones(data_range) * z0).double().cpu()
    xp = torch.from_numpy(np.ones(data_range) * x0).double().cpu()
    for k in range(1, kdmin):
        xx = xp
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
        xp = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * dt
        yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * dt
        zp = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * dt
        del kx1, kx2, kx3, ky1, ky2, ky3, kz1, kz2, kz3
    return xp, yp, zp


def main():
    R1, R2 = 6.00, 25.00
    C1, C2 = 2.80, 3.40
    x0, y0, z0 = 0.50, 4.00, 1.00
    NC, NL, NR = 600, 600, 600

    deltaC = (C2 - C1) / (NC - 1)
    deltaR = (R2 - R1) / (NR - 1)

    c_list = np.array([C1 + deltaC * i for i in range(0, 600)])
    r_list = np.array([R1 + deltaR * i for i in range(0, 300)])
    cr_period_list = np.transpose([np.repeat(r_list, len(c_list)), np.tile(c_list, len(r_list))])
    cr_period_list = np.append(cr_period_list, np.zeros((180000, 1)), axis=1)

    R = torch.from_numpy(cr_period_list[:, 0]).double().cpu()
    C = torch.from_numpy(cr_period_list[:, 1]).double().cpu()
    x, y, z = runge_kutta_4(R, C, x0, y0, z0, args.L, 180000, args.dt, int(args.tmin/args.dt))
    with open(args.out, 'a') as file:
        file.writelines('{}, {}, {} \n'.format((x[j]).cpu(), (y[j]).cpu(),
                                               (z[j]).cpu()) for j in range(0, 180000))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=1.00)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmin", type=int, default=1800)
    p.add_argument("--out", type=str, default="new_X_0_points.txt")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))