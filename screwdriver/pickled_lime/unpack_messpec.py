import numpy as np
import xml.etree.ElementTree as ET
from . import core_functions as cf
import os
import itertools
import dill as pickle
from psutil import virtual_memory
from . import spec_functions as sf
from .. import formatting


magic_bytes = b"Eg\x89\xab"
Nd = 4


def unpack_messpec(filelist_iter, filter_dict=None, loc="."):
    data = {}
    file_count = 0

    emergency_dumps = 0
    max_memory = virtual_memory().total / 30
    file_size = os.path.getsize(filelist_iter[0])
    check_interval = max_memory // file_size

    print("reading limes")
    for filename in filelist_iter:
        if (file_count := file_count + 1) % check_interval == 0:
            print(f"reading lime {file_count}")
            if cf.get_obj_size(data) > max_memory:
                print("Emergency dumping data")
                emergency_dumps += 1
                for attr, output in data.items():
                    out_dir = (
                        loc
                        + f"/messpec/{attr[0]}/{attr[1]}/{attr[2]}/"
                        + f"/{attr[3]}/{attr[4]}/"
                    )

                    os.system(f"mkdir -p {out_dir}")
                    out_name = f"messpec_{attr[5]}.pickle.temp{emergency_dumps}"
                    with open(out_dir + out_name, "wb") as file_out:
                        pickle.dump(np.array(output), file_out)

                data = {}

        file_in = open(filename.strip(), "rb")
        head, record = cf.read_record(file_in)
        if head[:4] != magic_bytes:
            raise IOError("Record header missing magic bytes.")

        if not head[16:].startswith(b"qcdsfDir"):
            raise IOError("Missing qcdsfDir record")

        tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
        root = tree.getroot()

        latt_size, latt_size_str = sf.read_latt_size(root)

        num_mom, mom_list = sf.read_momentum(root, Nd)

        head, record = cf.read_record(file_in)

        while head != b"":
            if not head[16:].startswith(b"meta-xml"):
                raise IOError("Expecting meta-xml record")
            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

            κ_str = sf.read_kappa(root)

            ferm_act_string = sf.read_ferm_act(root)

            source_sink_string = sf.read_source_sink(root)

            head, record = cf.read_record(file_in)
            if not head[16:].startswith(b"mesons-bin"):
                raise IOError("Expecting mesons-bin record")

            record = np.frombuffer(record, ">f8").reshape(
                16, 16, num_mom, latt_size[3], 2
            )

            for n, p in enumerate(mom_list):
                mom_str = formatting.format_mom(p)

                for γ1, γ2 in itertools.product(range(16), range(16)):

                    γ_str = f"{γ_names[γ1]}-{γ_names[γ2]}"

                    record_sliced = record[γ1, γ2, n]

                    attribute_list = tuple(
                        [
                            latt_size_str,
                            ferm_act_string,
                            κ_str,
                            source_sink_string,
                            mom_str,
                            γ_str,
                        ]
                    )
                    read_bool = cf.read_filter_dict(
                        filter_dict, attribute_names, attribute_list
                    )
                    if read_bool:
                        if attribute_list in data:
                            data[attribute_list].append(record_sliced)
                        else:
                            data[attribute_list] = [record_sliced]

            head, record = cf.read_record(file_in)

    print("writing pickles")
    for attr, output in data.items():
        out_dir = loc + f"/messpec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}/{attr[4]}/"
        os.makedirs(out_dir, exist_ok=True)
        out_data = []
        if emergency_dumps > 0:
            temp_name = f"messpec_{attr[5]}.pickle"
            for ed in range(emergency_dumps):
                with open(out_dir + temp_name + f".temp{ed+1}", "rb") as file_in:
                    out_data.append(pickle.load(file_in))
                os.remove(out_dir + temp_name + f".temp{ed+1}")

        out_data.append(output)
        out_data = np.array(out_data)
        ncfg = len(out_data)
        out_name = f"messpec_{attr[5]}_{ncfg}cfgs.pickle"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(out_data, file_out)

    return


attribute_names = [
    "lattice_size",
    "fermion_action",
    "kappa",
    "source_sink",
    "momentum",
    "gamma",
]

γ_names = [
    "gI",
    "g0",
    "g1",
    "g01",
    "g2",
    "g02",
    "g12",
    "g53",
    "g3",
    "g03",
    "g13",
    "g25",
    "g23",
    "g51",
    "g05",
    "g5",
]
