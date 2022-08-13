import numpy as np
import xml.etree.ElementTree as ET
from .. import core_functions as cf
import os
import pickle
from psutil import virtual_memory
from .. import readers
from .. import names


def read_barspec_qcdsf(file):
    head, record = cf.read_record(file)

    if not head[16:].startswith(b"qcdsfDir"):
        raise IOError("Missing qcdsfDir record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()

    lat_size, lat_size_str = readers.read_lat_size(root)

    Nd = len(lat_size)

    time_rev = readers.read_time_rev(root)

    num_mom, mom_list = readers.read_momentum(root, Nd)

    return Nd, time_rev, lat_size, lat_size_str, num_mom, mom_list


def read_baryons_meta(file):
    head, record = cf.read_record(file)
    if head == b"":
        return None, None, None, None, False
    if not head[16:].startswith(b"meta-xml"):
        raise IOError("Expecting meta-xml record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()
    has_third = readers.read_has_third(root)

    baryon_number = readers.read_baryon_number(root)

    κ_string = readers.read_kappa(root, has_third)

    ferm_act_string = readers.read_ferm_act(root)

    source_sink_string = readers.read_source_sink(root)

    return baryon_number, κ_string, ferm_act_string, source_sink_string, True


def baryons_bin_slicer(
    result,
    record,
    in_time_rev,
    lat_size,
    lat_size_string,
    num_mom,
    mom_list,
    baryon_number,
    κ_string,
    ferm_act_string,
    source_sink_string,
):
    record = np.frombuffer(record, ">f8").reshape(
        baryon_number, num_mom, lat_size[3], 2
    )

    for n, p in enumerate(mom_list):
        mom_string = format_mom(p)

        for b in range(baryon_number):
            bar_string = names.baryon_names[b]
            if in_time_rev:
                bar_string += "_trev"

            record_sliced = record[b, n]

            attribute_list = tuple(
                [
                    lat_size_string,
                    ferm_act_string,
                    κ_string,
                    source_sink_string,
                    mom_string,
                    bar_string,
                ]
            )
            result[attribute_list] = record_sliced
    return result


def read_baryons_bin(
    file,
    repeat,
    result,
    in_time_rev,
    lat_size,
    lat_size_string,
    num_mom,
    mom_list,
    baryon_number,
    κ_string,
    ferm_act_string,
    source_sink_string,
):

    if not repeat:
        return result

    head, record = cf.read_record(file)

    if in_time_rev:
        if not head[16:].startswith(b"baryons-trev-bin"):
            raise IOError("Expecting baryons-trev-bin record")
    else:
        if not head[16:].startswith(b"baryons-bin"):
            raise IOError("Expecting baryons-bin record")

    result = baryons_bin_slicer(
        result,
        record,
        in_time_rev,
        lat_size,
        lat_size_string,
        num_mom,
        mom_list,
        baryon_number,
        κ_string,
        ferm_act_string,
        source_sink_string,
    )
    return result


def read_barspec(file, *args, **kwargs):
    Nd, time_rev, lat_size, lat_size_string, num_mom, mom_list = read_barspec_qcdsf(
        file
    )
    result, repeat = {}, True
    while repeat:
        (
            baryon_number,
            κ_string,
            ferm_act_string,
            source_sink_string,
            repeat,
        ) = read_baryons_meta(file)
        result = read_baryons_bin(
            file,
            repeat,
            result,
            False,
            lat_size,
            lat_size_string,
            num_mom,
            mom_list,
            baryon_number,
            κ_string,
            ferm_act_string,
            source_sink_string,
        )
        if time_rev:
            result = read_baryons_bin(
                file,
                repeat,
                result,
                True,
                lat_size,
                lat_size_string,
                num_mom,
                mom_list,
                baryon_number,
                κ_string,
                ferm_act_string,
                source_sink_string,
            )
    return result


def emergency_write_barspec(data, loc, emergency_dumps):
    for attr, output in data.items():
        out_dir = (
            loc + f"/barspec/{attr[0]}/{attr[1]}/{attr[2]}/{attr[3]}" + f"/{attr[4]}/"
        )

        os.system(f"mkdir -p {out_dir}")

        out_name = f"barspec_{attr[5]}" + f".pickle.temp{emergency_dumps}"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(np.array(output), file_out)


def write_barspec(data, loc, emergency_dumps):
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
        out_name = f"barspec_{attr[5]}.pickle"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(out_data, file_out)


def format_mom(mom):
    return "p" + "".join([f"{mom_i:+d}" for mom_i in mom])
