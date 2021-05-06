# %%
import numpy as np


def ReadRecord(file):
    head = file.read(32)
    if head == b"":
        return b"", b""
    file.read(14*8)
    record_size = np.frombuffer(head[8:16], dtype='>i8')[0]
    record = file.read(record_size)
    padding_size = (8 - record_size % 8) % 8
    file.read(padding_size)
    return head, record


def Cartesian(n, lat_size):
    coord = []
    for i in range(len(lat_size)):
        coord.append(n % lat_size[i])
        n //= lat_size[i]
    return coord


def CountMom(mom2_max, Nd, mom_offset=0):
    L = int(np.ceil(np.sqrt(mom2_max)))
    mom_vol = 1
    mom_size = []
    mom_list = []
    for mu in range(Nd-1):
        mom_vol *= (2*L)+1
        mom_size.append((2*L)+1)
    num_mom = 0
    for n in range(mom_vol):
        mom = Cartesian(n, mom_size)
        mom2 = 0
        for i in range(3):
            mom[i] -= L
            mom2 += mom[i]**2
        if mom2 > mom2_max:
            pass
        else:
            num_mom += 1
            mom_list.append(mom)
    return num_mom, mom_list
