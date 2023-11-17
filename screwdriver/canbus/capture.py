import numpy as np
import pandas as pd
import base64

from . import tang


class Capture:
    def __init__(self):
        self.data = None
        self.ids = None
        self.binary = None
        self.integer = None
        self.tang = None

    def _binary_arr_helper(self, data):
        bin_arr = np.unpackbits(
            np.frombuffer(bytearray.fromhex(data[2:]), "uint8")
        ).astype(int)
        return np.concatenate((bin_arr, np.array([np.nan] * (64 - len(bin_arr)))))

    def binary_arr(self):
        self.binary = pd.DataFrame(
            np.array(list(map(self._binary_arr_helper, self.data["Hex"].to_numpy()))),
            columns=[f"Bit {i}" for i in range(64)],
        )
        self.binary["ID"] = self.data["ID"]
        self.binary["Time Seconds"] = self.data["Time Seconds"]
        self.binary["Time Epoch"] = self.data["Time Epoch"]
        self.binary["Timestamp"] = self.data["Timestamp"]

    def calc_bit_avg(self, mask=None):
        result = []
        if self.binary is None:
            self.binary_arr()
        if mask is None:
            data = self.binary[[f"Bit {i}" for i in range(64)]]
            ids = self.data["ID"]
        else:
            data = self.binary[mask]
            ids = self.data["ID"][mask]
        for id in self.ids:
            avg = data[(ids == id).to_numpy()].mean(axis=0)
            result.append([id, *avg])
        self.bit_avg = pd.DataFrame(
            result, columns=["ID", *[f"Bit {i}" for i in range(64)]]
        )
        return result

    def calc_bit_power_avg(self, power=1, mask=None):
        result = []
        if self.binary is None:
            self.binary_arr()
        if mask is None:
            data = self.binary
            ids = self.data["ID"]
        else:
            data = self.binary[mask]
            ids = self.data["ID"][mask]
        for id in self.ids:
            avg = np.power(
                np.power(data[(ids == id).to_numpy()], power).mean(axis=0), 1 / power
            )
            result.append([id, avg])
        return result

    def calc_tang(self, mask=None):
        result = tang.calc_tang(self, mask)
        return result

    def _integer_arr_helper(self, data):
        return np.concatenate(
            [
                np.array([int(a + b, 16) for a, b in zip(data[2::2], data[3::2])]),
                np.array([0] * (8 - (len(data[2:]) // 2))),
            ]
        )

    def integer_arr(self):
        self.integer = pd.DataFrame(
            np.array(list(map(self._integer_arr_helper, self.data["Hex"].to_numpy()))),
            columns=[f"Byte {i}" for i in range(8)],
        )
        self.integer["ID"] = self.data["ID"]
        self.integer["Timestamp"] = self.data["Timestamp"]

    def simple_matching_coeff(self, event_mask):
        event_mask = event_mask.astype(int)
        M_same = self.binary == event_mask[:, None]
        M_diff = self.binary != event_mask[:, None]
        result = []
        for id in self.ids:
            n_same = M_same[self.data["ID"] == id].sum(axis=0)
            n_diff = M_diff[self.data["ID"] == id].sum(axis=0)
            smc = n_same / (n_same + n_diff)
            result.append([id, *smc])
        result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
        return result

    def bin_pearson_coeff(self, event_mask):
        event_mask = event_mask.astype(int)
        if self.binary is None:
            self.binary_arr()
        result = []
        for id in self.ids:
            data = self.binary[self.data["ID"] == id][[f"Bit {i}" for i in range(64)]]
            mask = event_mask[self.data["ID"] == id]
            cov = np.dot(mask - mask.mean(), data - data.mean(axis=0)) / (
                mask.shape[0] - 1
            )

            corr = cov / np.sqrt(np.var(mask, ddof=1) * np.var(data, axis=0, ddof=1))
            corr[np.isnan(corr)] = 0
            result.append([id, *corr])
        result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
        return result

    def tang_pearson_coeff(self, event_mask):
        event_mask = event_mask.astype(int)
        if self.tang is None:
            self.calc_tang()
        result = []
        for id in self.ids:
            data = self.tang[self.data["ID"] == id][self.tang.columns[1:]].to_numpy()
            data[np.isnan(data)] = 0
            mask = event_mask[self.data["ID"] == id]
            cov = np.dot(mask - mask.mean(), data - data.mean(axis=0)) / (
                mask.shape[0] - 1
            )

            corr = cov / np.sqrt(np.var(mask, ddof=1) * np.var(data, axis=0, ddof=1))
            corr[np.isnan(corr)] = 0
            result.append([id, *corr])
        result = pd.DataFrame(result, columns=["ID", *[f"Bit {i}" for i in range(64)]])
        return result

    def rolling_tang(self, interval):
        return tang.rolling_tang(self, interval)

    def rolling_tang_binned(self, interval):
        return tang.rolling_tang_binned(self, interval)


class CaptureCanUtils(Capture):
    def __init__(self, file):
        super().__init__()
        self.data = self.read_candump(file)
        self.ids = np.sort(self.data["ID"].unique())

    def read_candump(self, logfile):
        result = []
        with open(logfile, "r") as f:
            for line in f:
                time, _, packet = line.rstrip("\n").split(" ")
                time = float(time.lstrip("(").rstrip(")"))
                can_id, can_data = packet.split("#")
                can_id = "0x" + can_id
                can_data = "0x" + can_data
                result.append([can_id, can_data, time])
        result = pd.DataFrame(result)
        result.columns = ["ID", "Hex", "Time Epoch"]
        result["Time Seconds"] = result["Time Epoch"] - result["Time Epoch"].min()
        result["Timestamp"] = pd.to_datetime(result["Time Epoch"], unit="s")
        return result


class CaptureCanKing(Capture):
    def __init__(self, file):
        super().__init__()
        self.data = self.read_canking(file)
        self.ids = np.sort(self.data["ID"].unique())
        self.binary_arr()
        self.integer_arr()

    def read_canking(self, file_loc):
        result = []
        with open(file_loc, "r") as file_in:
            line = file_in.readline()
            line = file_in.readline()
            for line in file_in:
                line_split = line.split()
                if line_split[0] == "Logging":
                    continue
                elif line_split[2] == "ErrorFrame":
                    continue

                can_id = "0x" + line_split[1]
                dlc = int(line_split[2])
                data = "0x" + "".join(line_split[3 : (3 + dlc)]) if dlc > 0 else "0x00"
                time = float(line_split[3 + dlc])

                result.append([can_id, data, time])
        result = pd.DataFrame(result)
        result.columns = ["ID", "Hex", "Time Epoch"]
        result["Time Seconds"] = result["Time Epoch"] - result["Time Epoch"].min()
        result["Timestamp"] = pd.to_datetime(result["Time Epoch"], unit="s")
        return result


class CaptureCSV(Capture):
    def __init__(self, file):
        super().__init__()
        self.data = self.read_csv(file)
        self.ids = np.sort(self.data["ID"].unique())
        # self.binary_arr()
        # self.integer_arr()

    def read_csv(self, file):
        result = pd.read_csv(file)
        result = result.rename(
            columns={
                "timestamp": "Time Epoch",
                "arbitration_id": "ID",
                "dlc": "DLC",
                "data": "Hex",
                "error": "Error",
                "extended": "Extended",
                "remote": "Remote",
            }
        )
        hex_chars = "abcdef"
        upper_map = str.maketrans(hex_chars, hex_chars.upper())
        result["Hex"] = "0x" + (
            result["Hex"]
            .apply(base64.b64decode)
            .apply(base64.b16encode)
            .apply(bytes.decode, args=("utf-8",))
        )
        result["ID"] = result["ID"].astype("string")
        result["ID int"] = result["ID"].apply(int, base=16)
        result["Timestamp"] = pd.to_datetime(result["Time Epoch"], unit="s")
        result["Time Seconds"] = result["Time Epoch"] - result["Time Epoch"].min()
        return result
