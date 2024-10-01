import pandas as pd
import pyproj
from scipy.optimize import curve_fit
import numpy as np


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def linear(x, a, b):
    return a * x + b


class GPS:
    def __init__(self, input_csv):
        self.data = pd.read_csv(input_csv, names=["Time", "Time2", "Lon", "Lat", "Alt"])

        latlon = pyproj.CRS("epsg:4326")
        xy = pyproj.CRS("epsg:28355")
        transformer = pyproj.Transformer.from_crs(latlon, xy)
        self.data[["x", "y"]] = self.data.apply(
            lambda row: transformer.transform(row["Lat"], row["Lon"]),
            result_type="expand",
            axis=1,
        )
        self.data["x"] -= self.data["x"].iloc[0]
        self.data["y"] -= self.data["y"].iloc[0]

    def vel_acc(self, n_points=3):
        indices = np.array([x - (n_points - 1) // 2 for x in list(range(n_points))])
        offset_i = -indices[0]
        offset_f = indices[-1]
        vel = np.zeros((len(self.data), 3))
        acc = np.zeros((len(self.data), 4))
        for i in range(offset_i, len(self.data) - offset_f):
            temp = self.data.iloc[indices + i][["Time", "x", "y"]]
            temp["Time"] -= self.data["Time"].iloc[i]
            fit_x = curve_fit(quadratic, temp["Time"].to_numpy(), temp["x"].to_numpy())[
                0
            ]
            fit_y = curve_fit(quadratic, temp["Time"].to_numpy(), temp["y"].to_numpy())[
                0
            ]
            vel_mag = np.sqrt(fit_x[1] ** 2 + fit_y[1] ** 2)
            vel[i] = (fit_x[1], fit_y[1], vel_mag)
            acc_long = (fit_x[0] * fit_x[1] + fit_y[0] * fit_y[1]) / vel_mag
            acc_tran = (fit_x[0] * fit_y[1] - fit_y[0] * fit_x[1]) / vel_mag
            acc[i] = fit_x[0], fit_y[0], acc_long, acc_tran
        self.data[["vel_x", "vel_y", "vel"]] = vel
        self.data[["acc_x", "acc_y", "acc_long", "acc_tran"]] = acc
