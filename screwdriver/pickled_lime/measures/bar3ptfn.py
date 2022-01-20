# %%
import numpy as np
import xml.etree.ElementTree as ET
from .. import core_functions as cf
import os
import dill as pickle

from ... import formatting
from .. import readers

# Unpack bar3ptfn lime files into pickled arrays.
# transition_form: label masses by {kin}t{kout}_{ki}..., where kin and kout are
# masses of propagators before and after current insertion.
# if false and kin=kout, label as {kcur}_{ki}..., where kcur is the mass of the
# propagator on which current is inserted.
# simplify: if true, label masses by {ki}... for unique ki, i.e. only listing all
# unique masses with no distinguishing current from forward props.
# NOTE simplify overrides transition_form


def read_bar3ptfn_meta(file, simplify, transition_form):
    head, record = cf.read_record(file)
    if head[:4] != cf.magic_bytes:
        raise IOError("Record header missing magic bytes.")

    if not head[16:].startswith(b"meta-xml"):
        raise IOError("Missing meta-xml record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()

    forward_props = readers.read_forward_props_bar3ptfn(root)

    lat_size, lat_size_string = readers.read_lat_size_bar3ptfn(root)

    ferm_act_string = readers.read_ferm_act_bar3ptfn(root)

    source_sink_string = readers.read_source_sink_bar3ptfn(forward_props)

    num_seqsrc, seqsrc_type, seqsrc = readers.read_seqsrc_bar3ptfn(root, lat_size)

    κ_string = readers.read_kappa_bar3ptfn(
        root, seqsrc, forward_props, simplify, transition_form
    )

    num_mom, mom_list = readers.read_mom_bar3ptfn(root, len(lat_size))

    return (
        lat_size,
        lat_size_string,
        κ_string,
        num_mom,
        mom_list,
        num_seqsrc,
        seqsrc_type,
        ferm_act_string,
        source_sink_string,
    )


def read_bar3ptfn_xml(file):
    head, record = cf.read_record(file)

    if not head[16:].startswith(b"bar3ptfn-xml"):
        raise IOError("Expecting bar3ptfn-xml record")

    tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
    root = tree.getroot()

    deriv = int(root.find("deriv").text)
    num_form_fac = 16 * (4 ** deriv)
    return deriv, num_form_fac


def read_bar3ptfn_bin(
    result,
    file,
    lat_size,
    lat_size_string,
    κ_string,
    num_mom,
    mom_list,
    num_seqsrc,
    seqsrc_type,
    ferm_act_string,
    source_sink_string,
    deriv,
    num_form_fac,
):
    head, record = cf.read_record(file)

    if not head[16:].startswith(b"bar3ptfn-bin"):
        raise IOError("Expecting bar3ptfn-bin record")

    record = np.frombuffer(record, ">f8").reshape(
        num_seqsrc, num_form_fac, num_mom, lat_size[3], 2
    )

    for n_seq, seq in enumerate(seqsrc_type):
        κ_str = κ_string[n_seq]

        for n_form_fac in range(num_form_fac):
            form_fac_str = formatting.format_form_fac(n_form_fac, deriv)

            for n_mom, mom in enumerate(mom_list):
                mom_str = formatting.format_mom(mom)

                record_sliced = record[n_seq, n_form_fac, n_mom]

                attribute_list = tuple(
                    [
                        lat_size_string,
                        ferm_act_string,
                        κ_str,
                        source_sink_string,
                        seq,
                        mom_str,
                        form_fac_str,
                    ]
                )
                result[attribute_list] = record_sliced
    return result


def read_bar3ptfn(file, simplify, transition_form):
    (
        lat_size,
        lat_size_string,
        κ_string,
        num_mom,
        mom_list,
        num_seqsrc,
        seqsrc_type,
        ferm_act_string,
        source_sink_string,
    ) = read_bar3ptfn_meta(file, simplify, transition_form)
    deriv, num_form_fac = read_bar3ptfn_xml(file)
    result = {}
    result = read_bar3ptfn_bin(
        result,
        file,
        lat_size,
        lat_size_string,
        κ_string,
        num_mom,
        mom_list,
        num_seqsrc,
        seqsrc_type,
        ferm_act_string,
        source_sink_string,
        deriv,
        num_form_fac,
    )
    return result


def emergency_write_bar3ptfn(data, loc, emergency_dumps):
    for attr, output in data.items():
        out_dir = (
            loc
            + f"/bar3ptfn/{attr[0]}/{attr[1]}/{attr[2]}/{attr[4]}"
            + f"/{attr[3]}/{attr[5]}/"
        )

        os.system(f"mkdir -p {out_dir}")

        out_name = f"bar3ptfn_{attr[6]}" + f".pickle.temp{emergency_dumps}"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(np.array(output), file_out)


def write_bar3ptfn(data, loc, emergency_dumps):
    for attr, output in data.items():
        out_dir = (
            loc
            + f"/bar3ptfn/{attr[0]}/{attr[1]}/{attr[2]}/{attr[4]}"
            + f"/{attr[3]}/{attr[5]}/"
        )
        os.makedirs(out_dir, exist_ok=True)
        if emergency_dumps > 0:
            out_data = []
            temp_name = f"bar3ptfn_{attr[6]}.pickle"
            for ed in range(emergency_dumps):
                with open(out_dir + temp_name + f".temp{ed+1}", "rb") as file_in:
                    out_data.append(pickle.load(file_in))
                os.remove(out_dir + temp_name + f".temp{ed+1}")
            if output != None:
                out_data.append(output)
            out_data = np.concatenate(out_data, axis=0)
        else:
            out_data = output
        out_data = np.array(out_data)
        ncfg = len(out_data)
        out_name = f"bar3ptfn_{attr[6]}_{ncfg}cfgs.pickle"
        with open(out_dir + out_name, "wb") as file_out:
            pickle.dump(out_data, file_out)

