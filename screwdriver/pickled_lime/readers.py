from . import core_functions as cf
import numpy as np

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


def read_lat_size(root):
    latt_size = [
        int(s)
        for s in root.find("ProgramInfo")
        .find("Setgeom")
        .find("latt_size")
        .text.split(" ")
    ]

    latt_size_str = format_lat_size(latt_size)
    return latt_size, latt_size_str


def read_momentum(root, Nd):
    mom2_max = int(root.find("Input").find("Param").find("mom2_max").text)
    num_mom, mom_list = cf.count_mom(mom2_max, Nd)
    return num_mom, mom_list


def read_kappa(root, has_third=False):
    κ1 = float(
        root.find("Forward_prop_headers")
        .find("First_forward_prop")
        .find("ForwardProp")
        .find("FermionAction")
        .find("Kappa")
        .text
    )
    κ2 = float(
        root.find("Forward_prop_headers")
        .find("Second_forward_prop")
        .find("ForwardProp")
        .find("FermionAction")
        .find("Kappa")
        .text
    )

    κ_str = (
        "k"
        + f"{κ1:.6f}".lstrip("0").replace(".", "p")
        + "k"
        + f"{κ2:.6f}".lstrip("0").replace(".", "p")
    )

    if has_third:
        κ3 = float(
            root.find("Forward_prop_headers")
            .find("Third_forward_prop")
            .find("ForwardProp")
            .find("FermionAction")
            .find("Kappa")
            .text
        )
        κ_str += "k" + f"{κ3:.6f}".lstrip("0").replace(".", "p")
    return κ_str


def read_ferm_act(root):

    ferm_act_string = (
        root.find("Forward_prop_headers")
        .find("First_forward_prop")
        .find("ForwardProp")
        .find("FermionAction")
        .find("FermAct")
        .text.lower()
    )
    if ferm_act_string == "clover":
        clover_coeff = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
            .find("ForwardProp")
            .find("FermionAction")
            .find("clovCoeff")
            .text.lstrip("0")
            .replace(".", "p")
        )
        ferm_act_string += "_" + clover_coeff
    return ferm_act_string


def read_source_sink(root):
    # Chroma throws an error if both props have different smearing, need
    # only check one each at source and sink

    source_string = root.find("SourceSinkType").find("source_type_1").text.lower()
    sink_string = root.find("SourceSinkType").find("sink_type_1").text.lower()

    if source_string == "shell_source":
        source_kind = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
            .find("PropSource")
            .find("Source")
            .find("SmearingParam")
            .find("wvf_kind")
            .text.lower()
        )
        source_param = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
            .find("PropSource")
            .find("Source")
            .find("SmearingParam")
            .find("wvf_param")
            .text.lstrip("0")
            .replace(".", "p")
        )
        source_intparam = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
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
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
            .find("PropSink")
            .find("Sink")
            .find("SmearingParam")
            .find("wvf_kind")
            .text.lower()
        )
        sink_param = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
            .find("PropSink")
            .find("Sink")
            .find("SmearingParam")
            .find("wvf_param")
            .text.lstrip("0")
            .replace(".", "p")
        )
        sink_intparam = (
            root.find("Forward_prop_headers")
            .find("First_forward_prop")
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


def read_latt_size_bar3ptfn(root):
    latt_size = [
        int(s)
        for s in root.find("bar3ptfn")
        .find("ProgramInfo")
        .find("Setgeom")
        .find("latt_size")
        .text.split(" ")
    ]
    latt_size_str = format_lat_size(latt_size)
    return latt_size, latt_size_str


def read_mom_bar3ptfn(root, Nd):
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
        κ_string = ["".join([format_kappa(k) for k in x]) for x in κ_simplified]

    elif (transition_form) or (~all([x == current_κ_in for x in current_κ_out])):
        κ_string = [
            format_kappa(current_κ_in)
            + "t"
            + format_kappa(x)
            + "_"
            + "".join([format_kappa(y) for y in forward_κ])
            for x in current_κ_out
        ]

    else:
        κ_string = [
            "c" + format_kappa(x) + "_" + "".join([format_kappa(y) for y in forward_κ])
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


def read_time_rev(root):
    return root.find("Input").find("Param").find("time_rev").text == "true"


def read_has_third(root):
    return not (root.find("Forward_prop_headers").find("Third_forward_prop") == None)


def read_baryon_number(root):
    return int(root.find("baryon-number").text)


def read_lat_size_bar3ptfn(root):
    lat_size = [
        int(s)
        for s in root.find("bar3ptfn")
        .find("ProgramInfo")
        .find("Setgeom")
        .find("latt_size")
        .text.split(" ")
    ]
    lat_size_str = format_lat_size(lat_size)
    return lat_size, lat_size_str


def read_mom_bar3ptfn(root, Nd):
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
        κ_string = ["".join([format_kappa(k) for k in x]) for x in κ_simplified]

    elif (transition_form) or (~all([x == current_κ_in for x in current_κ_out])):
        κ_string = [
            format_kappa(current_κ_in)
            + "t"
            + format_kappa(x)
            + "_"
            + "".join([format_kappa(y) for y in forward_κ])
            for x in current_κ_out
        ]

    else:
        κ_string = [
            "c" + format_kappa(x) + "_" + "".join([format_kappa(y) for y in forward_κ])
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


def read_filter_dict(filter_dict, attribute_names, attribute_list, **kwargs):
    if filter_dict == None:
        return True
    for name, val in zip(attribute_names, attribute_list):
        if name in filter_dict:
            read_bool = (
                hasattr(filter_dict[name], "__getitem__") and val in filter_dict[name]
            ) or (filter_dict[name] == val)
            if not read_bool:
                return False
        else:
            pass
    return True


def format_kappa(κ):
    return "k" + f"{κ:.6f}".lstrip("0").replace(".", "p")


def format_lat_size(*args, **kwargs):
    if args == ():
        return f"{kwargs['Ns']}x{kwargs['Nt']}"
    elif len(args) == 1 and type(args[0]) == list:
        temp = sorted(set(args[0]))
        return "x".join(str(x) for x in temp)
    elif np.logical_and([type(x) == int for x in args]):
        return "x".join(str(x) for x in args)
    else:
        raise ValueError(
            "input must be a dict with Ns and Nt keys, a single list of dimensions, or 1 int input per distinct dimension"
        )
