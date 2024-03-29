import numpy as np
import xml.etree.ElementTree as ET
from .. import core_functions as cf
import os
import itertools
import pickle
from psutil import virtual_memory
from .. import readers
from .. import names


def read_messpec_qcdsf(file):
    head, record = cf.read_record(file)

    if not head[16:].startswith(b"qcdsfDir"):
        raise IOError("Missing qcdsfDir record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()

    lat_size, lat_size_str = readers.read_lat_size(root)

    Nd = len(lat_size)

    num_mom, mom_list = readers.read_momentum(root, Nd)

    return Nd, lat_size, lat_size_str, num_mom, mom_list


def read_mesons_meta(file):
    head, record = cf.read_record(file)
    if head == b"":
        return None, None, None, False
    if not head[16:].startswith(b"meta-xml"):
        raise IOError("Expecting meta-xml record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()
    has_third = readers.read_has_third(root)

    κ_string = readers.read_kappa(root, has_third)

    ferm_act_string = readers.read_ferm_act(root)

    source_sink_string = readers.read_source_sink(root)

    return κ_string, ferm_act_string, source_sink_string, True


def mesons_bin_slicer(
    result,
    record,
    lat_size,
    lat_size_string,
    num_mom,
    mom_list,
    κ_string,
    ferm_act_string,
    source_sink_string,
):
    record = np.frombuffer(record, ">f8").reshape(16, 16, num_mom, lat_size[3], 2)

    for n, p in enumerate(mom_list):
        mom_string = format_mom(p)

        for γ1, γ2 in itertools.product(range(16), range(16)):
            γ_string = f"{names.γ_names[γ1]}-{names.γ_names[γ2]}"

            record_sliced = record[γ1, γ2, n]

            attribute_list = tuple(
                [
                    lat_size_string,
                    ferm_act_string,
                    κ_string,
                    source_sink_string,
                    mom_string,
                    γ_string,
                ]
            )
            result[attribute_list] = record_sliced
    return result


def read_mesons_bin(
    file,
    repeat,
    result,
    lat_size,
    lat_size_string,
    num_mom,
    mom_list,
    κ_string,
    ferm_act_string,
    source_sink_string,
):

    if not repeat:
        return result

    head, record = cf.read_record(file)

    if not head[16:].startswith(b"mesons-bin"):
        raise IOError("Expecting mesons-bin record")

    result = mesons_bin_slicer(
        result,
        record,
        lat_size,
        lat_size_string,
        num_mom,
        mom_list,
        κ_string,
        ferm_act_string,
        source_sink_string,
    )
    return result


def read_messpec(file, *args, **kwargs):
    Nd, lat_size, lat_size_string, num_mom, mom_list = read_messpec_qcdsf(file)
    result, repeat = {}, True
    while repeat:
        κ_string, ferm_act_string, source_sink_string, repeat = read_mesons_meta(file)
        result = read_mesons_bin(
            file,
            repeat,
            result,
            lat_size,
            lat_size_string,
            num_mom,
            mom_list,
            κ_string,
            ferm_act_string,
            source_sink_string,
        )
    return result


def emergency_write_messpec(data, loc, emergency_dumps):
    for attr, output in data.items():
        out_dir = (
            loc + f"/messpec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}" + f"/{attr[4]}/"
        )

        os.system(f"mkdir -p {out_dir}")

        out_name = f"messpec_{attr[5]}" + f".pickle.temp{emergency_dumps}"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(np.array(output), file_out)


def write_messpec(data, loc, emergency_dumps):
    for attr, output in data.items():
        out_dir = loc + f"/messpec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}/{attr[4]}/"
        os.makedirs(out_dir, exist_ok=True)
        if emergency_dumps > 0:
            out_data = []
            temp_name = f"messpec_{attr[5]}.pickle"
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
        out_name = f"messpec_{attr[5]}.pickle"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(out_data, file_out)


def format_mom(mom):
    return "p" + "".join([f"{mom_i:+d}" for mom_i in mom])
