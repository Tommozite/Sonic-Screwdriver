# %%
import numpy as np
import xml.etree.ElementTree as ET
from . import core_functions as cf
import os
import dill as pickle
import collections


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


def unpack_bar3ptfn(filelist_iter, loc="", transition_form=False, simplify=False):
    data = rec_dd()
    data_trev = rec_dd()

    for filename in filelist_iter:
        file_in = open(filename, "rb")
        head, record = cf.ReadRecord(file_in)
        if head[:4] != magic_bytes:
            raise IOError("Record header missing magic bytes.")

        if not head[16:].startswith(b"meta-xml"):
            raise IOError("Missing meta-xml record")

        tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
        root = tree.getroot()
        latt_size = [
            int(s)
            for s in root.find("bar3ptfn")
            .find("ProgramInfo")
            .find("Setgeom")
            .find("latt_size")
            .text.split(" ")
        ]
        seen = set()
        latt_size_str = "x".join(
            str(x) for x in latt_size if not (x in seen or seen.add(x))
        )
        deriv_max = int(
            root.find("bar3ptfn").find("Input").find("Param").find("deriv").text
        )

        mom2_max = int(
            root.find("bar3ptfn").find("Input").find("Param").find("mom2_max").text
        )
        num_mom, mom_list = cf.CountMom(mom2_max, Nd)

        seqsrc = (
            root.find("bar3ptfn")
            .find("Wilson_3Pt_fn_measurements")
            .find("Sequential_source")
            .findall("elem")
        )

        num_seqsrc = len(seqsrc)

        seqsrc_type = [
            x.find("SequentialProp_record_info")
            .find("SequentialProp")
            .find("SeqSource")
            .find("SeqSource")
            .find("SeqSourceType")
            .text
            for x in seqsrc
        ]

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
            κ_simplified = [
                set(forward_κ + [current_κ_in] + [x]) for x in current_κ_out
            ]
            κ_string = ["".join([FormatKappa(k) for k in x]) for x in κ_simplified]

        elif (transition_form) or (~all([x == current_κ_in for x in current_κ_out])):
            κ_string = [
                FormatKappa(current_κ_in)
                + "t"
                + FormatKappa(x)
                + "_"
                + "".join([FormatKappa(y) for y in forward_κ])
                for x in current_κ_out
            ]

        else:
            κ_string = [
                "c"
                + FormatKappa(x)
                + "_"
                + "".join([FormatKappa(y) for y in forward_κ])
                for x in current_κ_out
            ]

        ferm_act_string = (
            root.find("bar3ptfn")
            .find("Propagator_record_info")
            .find("Propagator")
            .find("ForwardProp")
            .find("FermionAction")
            .find("FermAct")
            .text.lower()
        )
        if ferm_act_string == "clover":
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
            ferm_act_string += "_" + clover_coeff

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
                f"{SourceNames(source_string)}"
                + f"_{SmearingNames(source_kind)}"
                + f"_{source_param}_{source_intparam}"
            )
        else:
            source_string = SourceNames(source_string)

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
                f"{SinkNames(sink_string)}"
                + f"_{SmearingNames(sink_kind)}"
                + f"_{sink_param}_{sink_intparam}"
            )
        else:
            sink_string = SinkNames(sink_string)
        source_sink_string = f"{source_string}-{sink_string}"

        head, record = cf.ReadRecord(file_in)

        if not head[16:].startswith(b"bar3ptfn-xml"):
            raise IOError("Expecting bar3ptfn-xml record")

        tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
        root = tree.getroot()

        deriv = int(root.find("deriv").text)
        num_form_fac = 16 * (4 ** deriv)

        head, record = cf.ReadRecord(file_in)

        if not head[16:].startswith(b"bar3ptfn-bin"):
            raise IOError("Expecting bar3ptfn-bin record")

        record = np.frombuffer(record, ">f8").reshape(
            num_seqsrc, num_mom, num_form_fac, latt_size[3], 2
        )

        for n_seq, seq in enumerate(seqsrc_type):
            κ_str = κ_string[n_seq]

            for n_form_fac in range(num_form_fac):
                form_fac_str = FormatFormFac(n_form_fac, deriv)

                for n_mom, mom in enumerate(mom_list):
                    mom_str = FormatMom(mom)

                    record_sliced = record[n_seq, n_mom, n_form_fac]

                    if (
                        type(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][seq][mom_str][form_fac_str]
                        )
                        == collections.defaultdict
                        and len(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][seq][mom_str][form_fac_str].keys()
                        )
                        == 0
                    ):
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            seq
                        ][mom_str][form_fac_str] = [record_sliced]
                    else:
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            seq
                        ][mom_str][form_fac_str].append(record_sliced)

    for latt_size, lvl1 in data.items():
        for ferm_act, lvl2 in lvl1.items():
            for κ, lvl3 in lvl2.items():
                for source_sink, lvl4 in lvl3.items():
                    for seq, lvl5 in lvl4.items():
                        for p, lvl6 in lvl5.items():
                            out_dir = (
                                loc
                                + f"/bar3ptfn/{latt_size}/{ferm_act}/{κ}/{seq}"
                                + f"/{source_sink}/{p}/"
                            )

                            os.system(f"mkdir -p {out_dir}")

                            for form_fac, lvl7 in lvl6.items():
                                ncfg = len(lvl7)
                                out_name = f"bar3ptfn_{form_fac}_{ncfg}cfgs.pickle"
                                with open(out_dir + out_name, "wb") as file_out:
                                    pickle.dump(np.array(lvl7), file_out)

    return


def rec_dd():
    return collections.defaultdict(rec_dd)


def FormatKappa(κ):
    return "k" + f"{κ:.6f}".lstrip("0").replace(".", "p")


def FormatMom(mom):
    return "p" + "".join([f"{mom_i:+d}" for mom_i in mom])


def SourceNames(name):
    names = {"shell_source": "sh", "point_source": "pt"}
    return names[name]


def SinkNames(name):
    names = {"shell_sink": "sh", "point_sink": "pt"}
    return names[name]


def SmearingNames(name):
    names = {"gauge_inv_jacobi": "gij"}
    return names[name]


def FormatFormFac(form_fac_number, deriv):
    gamma_num = form_fac_number % 16
    gamma_str = γString(gamma_num)
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


def γString(n):
    names = [
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
    return names[n]
