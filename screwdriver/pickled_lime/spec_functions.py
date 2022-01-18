from .. import formatting
from . import core_functions as cf


def read_latt_size(root):
    latt_size = [
        int(s)
        for s in root.find("ProgramInfo")
        .find("Setgeom")
        .find("latt_size")
        .text.split(" ")
    ]

    latt_size_str = formatting.format_lat_size(latt_size)
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
