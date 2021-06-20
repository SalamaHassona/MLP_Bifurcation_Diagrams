import numpy as np
import time
import torch
import argparse


class DataGenerator():

    def __init__(self, labeled_data_file, data_util_file, threshold=23,
                 dt=0.01, tmin=1800, tmax=2000, L=1.0):
        self.L = L
        self.dt = dt
        self.kdmin = int(tmin/dt)
        self.kd = int(tmax/dt)+1

        R1, R2 = 6.00, 25.00
        C1, C2 = 2.80, 3.40
        x0, y0, z0 = 0.50, 4.00, 1.00
        NC, NL, NR = 600, 600, 600

        deltaC = (C2 - C1) / (NC - 1)
        deltaR = (R2 - R1) / (NR - 1)

        xyz_points_untrimmed = np.loadtxt(data_util_file, delimiter=',')
        xyz_points = np.array([[item[0], item[1], item[2]] for item in xyz_points_untrimmed], dtype=np.double)
        original_untrimmed = np.loadtxt(labeled_data_file, delimiter=',')
        original = np.array([[item[0], item[1], 0 if item[2] < threshold else 1] for item in original_untrimmed],
                            dtype=np.double)
        original = np.concatenate((original, xyz_points), axis=1)
        idx_class_0 = np.where(original[:, 2] == 0)
        idx_class_1 = np.where(original[:, 2] == 1)
        len_class_0 = np.count_nonzero(original[:, 2] == 0)
        len_class_1 = np.count_nonzero(original[:, 2] == 1)
        class_0_points = original[idx_class_0]
        class_1_points = original[idx_class_1]
        np.random.shuffle(class_0_points)
        np.random.shuffle(class_1_points)

        if len_class_0 >= len_class_1:
            self.cr_point_list = np.concatenate((class_0_points[0:len_class_1, :], class_1_points), axis=0)
        else:
            self.cr_point_list = np.concatenate((class_0_points, class_1_points[0:len_class_0, :]), axis=0)

        np.random.shuffle(self.cr_point_list)

    def runge_kutta_4(self, R, C, x0, y0, z0, data_range, ts_nth_element=8):
        xp = torch.from_numpy(np.ones((data_range, self.kd - self.kdmin), dtype=np.double)).double().cpu()
        xp[:, 0] = x0
        yp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * y0
        zp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * z0
        for k in range(self.kdmin, self.kd - 1):
            xx = xp[:, k - self.kdmin]
            yy = yp
            zz = zp
            kx1 = (yy - xx * pow(zz, -2 / 3)) / self.L
            ky1 = (R + 1 - yy - R * xx) / (R * C)
            kz1 = xx * xx - zz
            x1 = xx + (self.dt * kx1) / 2
            y1 = yy + (self.dt * ky1) / 2
            z1 = zz + (self.dt * kz1) / 2
            kx2 = (y1 - x1 * pow(z1, -2 / 3)) / self.L
            ky2 = (R + 1 - y1 - R * x1) / (R * C)
            kz2 = x1 * x1 - z1
            del x1, y1, z1
            x2 = xx + (self.dt * kx2) / 2
            y2 = yy + (self.dt * ky2) / 2
            z2 = zz + (self.dt * kz2) / 2
            kx3 = (y2 - x2 * pow(z2, -2 / 3)) / self.L
            ky3 = (R + 1 - y2 - R * x2) / (R * C)
            kz3 = x2 * x2 - z2
            del x2, y2, z2
            x3 = xx + (self.dt * kx3)
            y3 = yy + (self.dt * ky3)
            z3 = zz + (self.dt * kz3)
            kx4 = (y3 - x3 * pow(z3, -2 / 3)) / self.L
            ky4 = (R + 1 - y3 - R * x3) / (R * C)
            kz4 = x3 * x3 - z3
            del x3, y3, z3
            xp[:, k - self.kdmin + 1] = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * self.dt
            yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * self.dt
            zp = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * self.dt
            del kx1, kx2, kx3, ky1, ky2, ky3, kz1, kz2, kz3
        del yp, zp
        xp = xp.numpy()[:, ::ts_nth_element]
        return xp

    def get_data(self, training_frac, ts_nth_element = 8):
        # data_len = 512
        data_len = self.cr_point_list.shape[0]
        training_len = int(data_len * training_frac)
        test_len = data_len - training_len
        R = torch.from_numpy(self.cr_point_list[0:data_len, 0]).double().cpu()
        C = torch.from_numpy(self.cr_point_list[0:data_len, 1]).double().cpu()
        x0 = torch.from_numpy(self.cr_point_list[0:data_len, 3]).double().cpu()
        y0 = torch.from_numpy(self.cr_point_list[0:data_len, 4]).double().cpu()
        z0 = torch.from_numpy(self.cr_point_list[0:data_len, 5]).double().cpu()

        x = self.runge_kutta_4(R, C, x0, y0, z0, data_len, ts_nth_element)
        data = np.column_stack((self.cr_point_list[:, 2], x))
        trainin_data = data[0:training_len, :]
        test_data = data[training_len:training_len + test_len, :]
        return trainin_data, test_data

    def get_data(self, batch_size, validation_frac, training_frac, ts_nth_element = 8):
        # data_len = 512
        data_len = self.cr_point_list.shape[0]
        training_len = (int(data_len * training_frac) // batch_size) * batch_size
        validation_len = (int(data_len * validation_frac) // batch_size) * batch_size
        test_len = (int(data_len - training_len - validation_len) // batch_size) * batch_size
        R = torch.from_numpy(self.cr_point_list[0:data_len, 0]).double().cpu()
        C = torch.from_numpy(self.cr_point_list[0:data_len, 1]).double().cpu()
        x0 = torch.from_numpy(self.cr_point_list[0:data_len, 3]).double().cpu()
        y0 = torch.from_numpy(self.cr_point_list[0:data_len, 4]).double().cpu()
        z0 = torch.from_numpy(self.cr_point_list[0:data_len, 5]).double().cpu()

        x = self.runge_kutta_4(R, C, x0, y0, z0, data_len, ts_nth_element)
        data = np.column_stack((self.cr_point_list[:, 2], x))
        trainin_data = data[0:training_len, :]
        valdiation_data = data[training_len:training_len + validation_len, :]
        test_data = data[training_len + validation_len:training_len + validation_len + test_len, :]
        return trainin_data, valdiation_data, test_data
