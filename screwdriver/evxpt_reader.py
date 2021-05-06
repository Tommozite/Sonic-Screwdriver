import numpy as np


def evxpt_read_dump(infile, measure=0):
    result = []
    with open(infile, "r") as file_in:
        for line in file_in:
            strip_line = line.rstrip()
            if len(strip_line) > 0:
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "E"
                    and strip_line[2] == "N"
                ):
                    for skip in range(2):
                        file_in.readline()
                    temp = file_in.readline().split()
                    t_size = int(temp[5])
                    x_size = int(temp[6])
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "R"
                    and strip_line[2] == "P"
                    and strip_line[4] == str(measure)
                ):
                    temp = file_in.readline().split()
                    while temp[0] != "nmeas":
                        temp = file_in.readline().split()
                    n_cfg = int(temp[2])
                    result = np.empty(shape=(t_size, n_cfg))
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "R"
                    and strip_line[2] == "D"
                    and strip_line[4] == str(measure)
                ):
                    for cfg in range(n_cfg):
                        for time in range(t_size):
                            temp = file_in.readline().split()
                            result[time, cfg] = float(temp[1])

    return result, n_cfg, t_size, x_size


def evxpt_read_data(infile, measure=0):
    result = []
    with open(infile, "r") as file_in:
        for line in file_in:
            strip_line = line.rstrip()
            if len(strip_line) > 0:
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "E"
                    and strip_line[2] == "N"
                ):
                    for skip in range(2):
                        file_in.readline()
                    temp = file_in.readline().split()
                    t_size = int(temp[5])
                    x_size = int(temp[6])
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "D"
                    and strip_line[2] == "A"
                    and strip_line[6] == str(measure)
                ):
                    temp = file_in.readline().split()
                    while not temp[0].startswith("+NU"):
                        temp = file_in.readline().split()
                    n_meas = int(temp[0][9:])
                    # result = xr.DataArray( np.empty(shape=(n_boot, n_par)), #dims=('boot','params'),
                    #                       coords={'boot':range(n_boot)})
                    result = np.empty(shape=(n_meas, 2))
                    temp = file_in.readline().split()
                    for meas in range(n_meas):
                        result[meas] = temp[1:]
                        temp = file_in.readline().split()

    return result, n_meas, t_size, x_size


def evxpt_read_fit(infile, measure=0):
    result = []
    with open(infile, "r") as file_in:
        for line in file_in:
            strip_line = line.rstrip()
            if len(strip_line) > 0:
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "E"
                    and strip_line[2] == "N"
                ):
                    for skip in range(2):
                        file_in.readline()
                    temp = file_in.readline().split()
                    t_size = int(temp[5])
                    x_size = int(temp[6])
                if (
                    strip_line[0] == "+"
                    and strip_line[1] == "F"
                    and strip_line[2] == "I"
                    and strip_line[5] == str(measure)
                ):
                    temp = file_in.readline().split()
                    n_par = 0
                    while not temp[0].startswith("+NU"):
                        if temp[0].startswith("par"):
                            n_par += 1
                        temp = file_in.readline().split()
                    n_boot = int(temp[0][9:])
                    # result = xr.DataArray( np.empty(shape=(n_boot, n_par)), #dims=('boot','params'),
                    #                       coords={'boot':range(n_boot)})
                    result = np.empty(shape=(n_boot, n_par))
                    temp = file_in.readline().split()
                    for boot in range(n_boot):
                        for par in range(n_par):
                            result[boot, par] = temp[(2 * par + 4)]
                        temp = file_in.readline().split()

    return result, n_boot, t_size, x_size
