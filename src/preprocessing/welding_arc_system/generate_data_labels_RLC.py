import numpy as np
import time
import torch
import argparse


def generate_data_labels(cr_period_list, x0, y0, z0, L, dt, kd, kdmin):
    R = torch.from_numpy(cr_period_list[:, 0]).double().cpu()
    C = torch.from_numpy(cr_period_list[:, 1]).double().cpu()
    xp = torch.from_numpy(np.ones(180000) * x0).double().cpu()
    yp = torch.from_numpy(np.ones(180000) * y0).double().cpu()
    zp = torch.from_numpy(np.ones(180000) * z0).double().cpu()
    yokr = torch.from_numpy(np.zeros((180000, 100))).double().cpu()
    l_bif = torch.from_numpy(np.zeros(180000)).long().cpu()
    l_okr = torch.from_numpy(np.zeros(180000)).long().cpu()
    # xml[0] = x0
    for k in range(1, kd + 1):
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
        x2 = xx + (dt * kx2) / 2
        y2 = yy + (dt * ky2) / 2
        z2 = zz + (dt * kz2) / 2
        kx3 = (y2 - x2 * pow(z2, -2 / 3)) / L
        ky3 = (R + 1 - y2 - R * x2) / (R * C)
        kz3 = x2 * x2 - z2
        x3 = xx + (dt * kx3)
        y3 = yy + (dt * ky3)
        z3 = zz + (dt * kz3)
        kx4 = (y3 - x3 * pow(z3, -2 / 3)) / L
        ky4 = (R + 1 - y3 - R * x3) / (R * C)
        kz4 = x3 * x3 - z3
        xp = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * dt
        yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * dt
        zp = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * dt
        # xml[k] = xp
        if k >= kdmin:
            r00 = (yy - 1.0) * (yp - 1.0)
            y00 = (xx - 1.0) - (yy - 1.0) * (xp - xx) / (yp - yy)
            condition = (r00 <= 0) & (y00 > 0)
            yokr[condition, l_bif[condition]] = y00[condition]
            l_bif[condition] += 1

    max_l_bif = max(l_bif).item()
    for i in range(0, max_l_bif):
        cond_i = l_bif == i + 1
        if yokr[cond_i, 0:i + 1].nelement() != 0:
            cond_i_idx = torch.where(cond_i)[0]
            periods, indices = yokr[cond_i, 0:i + 1].sort(descending=True)
            l_okr[cond_i] = 1
            tmp = periods[:, 0]
            for j in range(1, i + 1):
                cond_abs = torch.abs(periods[:, j] - tmp) >= dt
                cond_abs_idx = cond_i_idx[cond_abs]
                tmp[cond_abs] = periods[cond_abs, j]
                l_okr[cond_abs_idx] += 1
    return R, C, l_okr


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

    R, C, labels = generate_data_labels(cr_period_list, x0, y0, z0, args.L,
                                  args.dt, int(args.tmax/args.dt)+1, int(args.tmin/args.dt))

    with open(args.out, 'a') as file:
        file.writelines('{:.6f}, {:.6f}, {} \n'.format(R[j], C[j], labels[j]) for j in range(0, 180000))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=1.00)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmin", type=int, default=1000)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--out", type=str, default="labeled_data.txt")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))