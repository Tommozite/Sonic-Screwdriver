import numpy as np
import xml.etree.ElementTree as ET
from . import core_functions as cf
import os
import dill as pickle
from psutil import virtual_memory
from . import spec_functions as sf
from .. import formatting


magic_bytes = b"Eg\x89\xab"
Nd = 4


def unpack_barspec(filelist_iter, filter_dict=None, loc="."):
    data = {}
    data_trev = {}
    file_count = 0

    emergency_dumps = 0
    emergency_dumps_trev = 0
    max_memory = virtual_memory().total / 30
    file_size = os.path.getsize(filelist_iter[0])
    check_interval = max_memory // file_size

    print("reading limes")
    for filename in filelist_iter:
        if (file_count := file_count + 1) % check_interval == 0:
            if cf.get_obj_size(data) > max_memory:
                print("Emergency dumping data")
                emergency_dumps += 1
                for attr, output in data.items():
                    out_dir = (
                        loc
                        + f"/barspec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}"
                        + f"/{attr[4]}/"
                    )

                    os.system(f"mkdir -p {out_dir}")

                    out_name = f"barspec_{attr[5]}" + f".pickle.temp{emergency_dumps}"
                    with open(out_dir + out_name, "wb") as file_out:
                        pickle.dump(np.array(output), file_out)

                data = {}

            if cf.get_obj_size(data_trev) > max_memory:
                print("Emergency dumping data")
                emergency_dumps_trev += 1
                for attr, output in data_trev.items():
                    out_dir = (
                        loc
                        + f"/barspec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}"
                        + f"/{attr[4]}/"
                    )

                    os.system(f"mkdir -p {out_dir}")

                    out_name = (
                        f"barspec_{attr[5]}_timerev"
                        + f".pickle.temp{emergency_dumps_trev}"
                    )
                    with open(out_dir + out_name, "wb") as file_out:
                        pickle.dump(np.array(output), file_out)

                data_trev = {}

        file_in = open(filename, "rb")
        head, record = cf.read_record(file_in)
        if head[:4] != magic_bytes:
            raise IOError("Record header missing magic bytes.")

        if not head[16:].startswith(b"qcdsfDir"):
            raise IOError("Missing qcdsfDir record")
       
        tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
        root = tree.getroot()

        latt_size, latt_size_str = sf.read_latt_size(root)

        time_rev = read_time_rev(root)

        num_mom, mom_list = sf.read_momentum(root, Nd)

        head, record = cf.read_record(file_in)

        while head != b"":
            if not head[16:].startswith(b"meta-xml"):
                raise IOError("Expecting meta-xml record")
            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

            has_third = read_has_third(root)

            baryon_number = read_baryon_number(root)

            κ_str = sf.read_kappa(root, has_third)

            ferm_act_string = sf.read_ferm_act(root)

            source_sink_string = sf.read_source_sink(root)

            head, record = cf.read_record(file_in)

            if not head[16:].startswith(b"baryons-bin"):
                raise IOError("Expecting baryons-bin record")

            record = np.frombuffer(record, ">f8").reshape(
                baryon_number, num_mom, latt_size[3], 2
            )

            for n, p in enumerate(mom_list):
                mom_str = formatting.format_mom(p)

                for b in range(baryon_number):
                    bar_str = baryon_names[b]

                    record_sliced = record[b, n]

                    attribute_list = tuple(
                        [
                            latt_size_str,
                            ferm_act_string,
                            κ_str,
                            source_sink_string,
                            mom_str,
                            bar_str,
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

            if time_rev:
                if not head[16:].startswith(b"baryons-trev-bin"):
                    raise IOError("Expecting baryons-trev-bin record")

                record = np.frombuffer(record, ">f8").reshape(
                    baryon_number, num_mom, latt_size[3], 2
                )

                for n, p in enumerate(mom_list):
                    mom_str = formatting.format_mom(p)

                    for b in range(baryon_number):
                        bar_str = baryon_names[b]

                        record_sliced = record[b, n]
                        attribute_list = [
                            latt_size_str,
                            ferm_act_string,
                            κ_str,
                            source_sink_string,
                            mom_str,
                            bar_str,
                        ]
                        read_bool = cf.read_filter_dict(
                            filter_dict, attribute_names, attribute_list
                        )
                        if read_bool:
                            if attribute_list in data:
                                data_trev[attribute_list].append(record_sliced)
                            else:
                                data_trev[attribute_list] = [record_sliced]

                head, record = cf.read_record(file_in)

    for attr, output in data.items():
        out_dir = loc + f"/barspec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}/{attr[4]}/"
        os.makedirs(out_dir, exist_ok=True)
        if emergency_dumps > 0:
            out_data = []
            temp_name = f"barspec_{attr[5]}.pickle"
            for ed in range(emergency_dumps):
                with open(out_dir + temp_name + f".temp{ed+1}", "rb") as file_in:
                    out_data.append(pickle.load(file_in))
                os.remove(out_dir + temp_name + f".temp{ed+1}")
            out_data.append(output)
            out_data = np.concatenate(out_data, axis=0)
        else:
            out_data = output
        out_data = np.array(out_data)
        ncfg = len(out_data)
        out_name = f"barspec_{attr[5]}_{ncfg}cfgs.pickle"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(out_data, file_out)
    if time_rev:
        for attr, output in data_trev.items():
            out_dir = (
                loc + f"/barspec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}/{attr[4]}/"
            )
            if emergency_dumps_trev > 0:
                out_data = []
                temp_name = f"barspec_{attr[5]}_timerev.pickle"
                for ed in range(emergency_dumps):
                    with open(out_dir + temp_name + f".temp{ed+1}", "rb") as file_in:
                        out_data.append(pickle.load(file_in))
                    os.remove(out_dir + temp_name + f".temp{ed+1}")
                out_data.append(output)
                out_data = np.concatenate(out_data, axis=0)
            else:
                out_data = output
            out_data = np.array(out_data)
            ncfg = len(out_data)
            out_name = f"barspec_{attr[5]}_timerev_{ncfg}cfgs.pickle"
            with open(out_dir + out_name, "wb") as file_out:
                pickle.dump(out_data, file_out)

    return


attribute_names = [
    "lattice_size",
    "fermion_action",
    "kappa",
    "source_sink",
    "momentum",
    "baryon",
]

"Using a dict for readability"
baryon_names = {
    0: "noise_proton",
    1: "lambda8_rel",
    2: "delta_1",
    3: "sigma_polrelg4",
    4: "lambda_polrelg4",
    5: "delta_2",
    6: "sigma_polnr",
    7: "lambda8_nrup",
    8: "delta8",
    9: "nucleon_rel",
    10: "sigma_unpolrelg4",
    11: "nucleon_nr",
    12: "lambda_rel_naive",
    13: "xi_rel",
    14: "lambda_polrel_naive",
    15: "xi_polrel",
    16: "nucleon_star",
    17: "nucleon12x",
    18: "nucleon12y",
    19: "nucleon12",
    20: "nucleon34",
    21: "nucleon2",
    22: "lambda8_nrdown",
    23: "delta_half",
    24: "delta_mhalf",
    25: "delta_m3half",
    26: "sigma0_nf3",
    27: "lambda8_nf3",
    28: "lambda8-sigma0_nf3",
    29: "sigma0-lambda8_nf3",
    30: "delta_nf3",
}


def read_time_rev(root):
    return root.find("Input").find("Param").find("time_rev").text == "true"


def read_has_third(root):
    return not (root.find("Forward_prop_headers").find("Third_forward_prop") == None)


def read_baryon_number(root):
    return int(root.find("baryon-number").text)
