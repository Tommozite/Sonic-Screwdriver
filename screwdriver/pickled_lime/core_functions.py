# %%
import numpy as np
import gc
import sys
import xml.etree.ElementTree as ET
from . import readers
from . import names

magic_bytes = b"Eg\x89\xab"


def read_record(file):
    head = file.read(32)
    if head == b"":
        return b"", b""
    file.read(14 * 8)
    record_size = np.frombuffer(head[8:16], dtype=">i8")[0]
    record = file.read(record_size)
    padding_size = (8 - record_size % 8) % 8
    file.read(padding_size)
    if head[:4] != magic_bytes:
        raise IOError("Record header missing magic bytes.")
    return head, record


def cartesian(n, lat_size):
    coord = []
    for i in range(len(lat_size)):
        coord.append(n % lat_size[i])
        n //= lat_size[i]
    return coord


def count_mom(mom2_max, Nd, mom_offset=0):
    L = int(np.ceil(np.sqrt(mom2_max)))
    mom_vol = 1
    mom_size = []
    mom_list = []
    for mu in range(Nd - 1):
        mom_vol *= (2 * L) + 1
        mom_size.append((2 * L) + 1)
    num_mom = 0
    for n in range(mom_vol):
        mom = cartesian(n, mom_size)
        mom2 = 0
        for i in range(3):
            mom[i] -= L
            mom2 += mom[i] ** 2
        if mom2 > mom2_max:
            pass
        else:
            num_mom += 1
            mom_list.append(mom)
    return num_mom, mom_list


def source_names(name):
    names = {"shell_source": "sh", "point_source": "pt"}
    return names[name]


def sink_names(name):
    names = {"shell_sink": "sh", "point_sink": "pt"}
    return names[name]


def smearing_names(name):
    names = {"gauge_inv_jacobi": "gij", "gauge_inv_gaussian": "gig"}
    return names[name]


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {
            o_id: o
            for o_id, o in all_refr
            if o_id not in marked and not isinstance(o, type)
        }

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

