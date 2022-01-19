# %%
import numpy as np
import xml.etree.ElementTree as ET
from . import core_functions as cf
import os
import dill as pickle
import collections
from psutil import virtual_memory

from .. import formatting

magic_bytes = b"Eg\x89\xab"
Nd = 4

# Unpack bar3ptfn lime files into pickled arrays.
# transition_form: label masses by {kin}t{kout}_{ki}..., where kin and kout are
# masses of propagators before and after current insertion.
# if false and kin=kout, label as {kcur}_{ki}..., where kcur is the mass of the
# propagator on which current is inserted.
# simplify: if true, label masses by {ki}... for unique ki, i.e. only listing all
# unique masses with no distinguishing current from forward props.
# NOTE simplify overrides transition_form


def unpack_bar3ptfn(
    filelist_iter, loc=".", filter_dict=None, transition_form=False, simplify=False
):
    data = {}

    emergency_dumps = 0
    max_memory = 1000000000  # 10MB hardcoded for now #virtual_memory().total / 200
    file_size = os.path.getsize(filelist_iter[0])
    # check_interval = max_memory // file_size
    check_interval = len(filelist_iter) // 1
    print("reading limes")
    for file_count, filename in enumerate(filelist_iter):
        if (file_count + 1) % check_interval == 0:
            print(f"reading file {file_count+1}")
            size = cf.get_obj_size(data)
            if size > max_memory:
                print("Emergency dumping data")
                emergency_dumps += 1
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

                data = {key: None for key in data.keys()}

        with open(filename, "rb") as file_in:
            head, record = cf.read_record(file_in)
            if head[:4] != magic_bytes:
                raise IOError("Record header missing magic bytes.")

            if not head[16:].startswith(b"meta-xml"):
                raise IOError("Missing meta-xml record")

            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

            forward_props = read_forward_props_bar3ptfn(root)

            latt_size, latt_size_str = read_latt_size_bar3ptfn(root)

            ferm_act_string = read_ferm_act_bar3ptfn(root)

            source_sink_string = read_source_sink_bar3ptfn(forward_props)

            num_seqsrc, seqsrc_type, seqsrc = read_seqsrc_bar3ptfn(root, latt_size)

            κ_string = read_kappa_bar3ptfn(
                root, seqsrc, forward_props, simplify, transition_form
            )

            num_mom, mom_list = read_mom_bar3ptfn(root)

            head, record = cf.read_record(file_in)

            if not head[16:].startswith(b"bar3ptfn-xml"):
                raise IOError("Expecting bar3ptfn-xml record")

            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

            deriv = int(root.find("deriv").text)
            num_form_fac = 16 * (4 ** deriv)

            head, record = cf.read_record(file_in)

            if not head[16:].startswith(b"bar3ptfn-bin"):
                raise IOError("Expecting bar3ptfn-bin record")

            record = np.frombuffer(record, ">f8").reshape(
                num_seqsrc, num_form_fac, num_mom, latt_size[3], 2
            )

            for n_seq, seq in enumerate(seqsrc_type):
                κ_str = κ_string[n_seq]

                for n_form_fac in range(num_form_fac):
                    form_fac_str = format_form_fac(n_form_fac, deriv)

                    for n_mom, mom in enumerate(mom_list):
                        mom_str = formatting.format_mom(mom)

                        record_sliced = record[n_seq, n_form_fac, n_mom]

                        attribute_list = tuple(
                            [
                                latt_size_str,
                                ferm_act_string,
                                κ_str,
                                source_sink_string,
                                seq,
                                mom_str,
                                form_fac_str,
                            ]
                        )

                        read_bool = cf.read_filter_dict(
                            filter_dict, attribute_names, attribute_list
                        )

                        if read_bool:
                            if attribute_list in data and data[attribute_list] != None:
                                data[attribute_list].append(record_sliced)
                            else:
                                data[attribute_list] = [record_sliced]

    print("writing pickles")
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

    return


attribute_names = [
    "lattice_size",
    "fermion_action",
    "kappa",
    "source_sink",
    "sequential_source",
    "momentum",
    "form_factor",
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


def format_form_fac(form_fac_number, deriv):
    gamma_num = form_fac_number % 16
    gamma_str = γ_names[gamma_num]
    mu = (form_fac_number // 16) % 4
    nu = form_fac_number // 64
    if deriv == 0:
        result = gamma_str
    elif deriv == 1:
        result = f"mu{mu}_" + gamma_str
    elif deriv == 2:
        result = f"mu{mu}_nu{nu}_" + gamma_str
    else:
        raise ValueError(f"deriv = {deriv} not supported.")
    return result


def read_latt_size_bar3ptfn(root):
    latt_size = [
        int(s)
        for s in root.find("bar3ptfn")
        .find("ProgramInfo")
        .find("Setgeom")
        .find("latt_size")
        .text.split(" ")
    ]
    latt_size_str = formatting.format_lat_size(latt_size)
    return latt_size, latt_size_str


def read_mom_bar3ptfn(root):
    mom2_max = int(
        root.find("bar3ptfn").find("Input").find("Param").find("mom2_max").text
    )
    num_mom, mom_list = cf.count_mom(mom2_max, Nd)
    return num_mom, mom_list


def read_seqsrc_bar3ptfn(root, latt_size):
    seqsrc = (
        root.find("bar3ptfn")
        .find("Wilson_3Pt_fn_measurements")
        .find("Sequential_source")
        .findall("elem")
    )

    num_seqsrc = len(seqsrc)

    seqsrc_type = [
        x.find("seqsrc_type").text.lower()
        + "_"
        + γ_names[int(x.find("gamma_insertion").text)]
        + "_t"
        + str(
            (int(x.find("t_sink").text) - int(x.find("t_source").text)) % latt_size[3]
        )
        for x in seqsrc
    ]
    return num_seqsrc, seqsrc_type, seqsrc


def read_forward_props_bar3ptfn(root):
    forward_props = (
        root.find("bar3ptfn")
        .find("Wilson_3Pt_fn_measurements")
        .find("Sequential_source")
        .find("elem")
        .find("SequentialProp_record_info")
        .find("SequentialProp")
        .find("ForwardProps")
        .findall("elem")
    )
    return forward_props


def read_kappa_bar3ptfn(root, seqsrc, forward_props, simplify, transition_form):
    forward_κ = [
        float(x.find("ForwardProp").find("FermionAction").find("Kappa").text)
        for x in forward_props
    ]
    current_κ_in = float(
        root.find("bar3ptfn")
        .find("Propagator_record_info")
        .find("Propagator")
        .find("ForwardProp")
        .find("FermionAction")
        .find("Kappa")
        .text
    )

    current_κ_out = [
        float(
            x.find("SequentialProp_record_info")
            .find("SequentialProp")
            .find("SeqProp")
            .find("FermionAction")
            .find("Kappa")
            .text
        )
        for x in seqsrc
    ]
    if simplify:
        κ_simplified = [set(forward_κ + [current_κ_in] + [x]) for x in current_κ_out]
        κ_string = [
            "".join([formatting.format_kappa(k) for k in x]) for x in κ_simplified
        ]

    elif (transition_form) or (~all([x == current_κ_in for x in current_κ_out])):
        κ_string = [
            formatting.format_kappa(current_κ_in)
            + "t"
            + formatting.format_kappa(x)
            + "_"
            + "".join([formatting.format_kappa(y) for y in forward_κ])
            for x in current_κ_out
        ]

    else:
        κ_string = [
            "c"
            + formatting.format_kappa(x)
            + "_"
            + "".join([formatting.format_kappa(y) for y in forward_κ])
            for x in current_κ_out
        ]
    return κ_string


def read_ferm_act_bar3ptfn(root):
    ferm_act_string = (
        root.find("bar3ptfn")
        .find("Propagator_record_info")
        .find("Propagator")
        .find("ForwardProp")
        .find("FermionAction")
        .find("FermAct")
        .text.lower()
    )
    if ferm_act_string == "clover" or ferm_act_string == "clover_fh":
        clover_coeff = (
            root.find("bar3ptfn")
            .find("Propagator_record_info")
            .find("Propagator")
            .find("ForwardProp")
            .find("FermionAction")
            .find("clovCoeff")
            .text.lstrip("0")
            .replace(".", "p")
        )
        ferm_act_string = "clover_" + clover_coeff
    return ferm_act_string


def read_source_sink_bar3ptfn(forward_props):

    source_string = (
        forward_props[0]
        .find("PropSource")
        .find("Source")
        .find("SourceType")
        .text.lower()
    )
    sink_string = (
        forward_props[0].find("PropSink").find("Sink").find("SinkType").text.lower()
    )

    if source_string == "shell_source":
        source_kind = (
            forward_props[0]
            .find("PropSource")
            .find("Source")
            .find("SmearingParam")
            .find("wvf_kind")
            .text.lower()
        )
        source_param = (
            forward_props[0]
            .find("PropSource")
            .find("Source")
            .find("SmearingParam")
            .find("wvf_param")
            .text.lstrip("0")
            .replace(".", "p")
        )
        source_intparam = (
            forward_props[0]
            .find("PropSource")
            .find("Source")
            .find("SmearingParam")
            .find("wvfIntPar")
            .text
        )
        source_string = (
            f"{cf.source_names(source_string)}"
            + f"_{cf.smearing_names(source_kind)}"
            + f"_{source_param}_{source_intparam}"
        )
    else:
        source_string = cf.source_names(source_string)

    if sink_string == "shell_sink":
        sink_kind = (
            forward_props[0]
            .find("PropSink")
            .find("Sink")
            .find("SmearingParam")
            .find("wvf_kind")
            .text.lower()
        )
        sink_param = (
            forward_props[0]
            .find("PropSink")
            .find("Sink")
            .find("SmearingParam")
            .find("wvf_param")
            .text.lstrip("0")
            .replace(".", "p")
        )
        sink_intparam = (
            forward_props[0]
            .find("PropSink")
            .find("Sink")
            .find("SmearingParam")
            .find("wvfIntPar")
            .text
        )
        sink_string = (
            f"{cf.sink_names(sink_string)}"
            + f"_{cf.smearing_names(sink_kind)}"
            + f"_{sink_param}_{sink_intparam}"
        )
    else:
        sink_string = cf.sink_names(sink_string)
    source_sink_string = f"{source_string}-{sink_string}"
    return source_sink_string
