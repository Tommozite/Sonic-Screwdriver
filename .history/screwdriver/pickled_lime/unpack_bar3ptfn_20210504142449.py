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
# transition_form: label masses by {kin}t{kout}_{ki}_..., where kin and kout are
# masses of propagators before and after current insertion.
# if false and kin=kout, label as {kcur}_{ki}_..., where kcur is the mass of the
# propagator on which current is inserted.
# simplify: if true, label masses by {ki}_... for unique ki, i.e. only listing all
# unique masses with no distinguishing current from forward props.


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

        seqsrc = ( root.find("bar3ptfn")
            .find("Wilson_3Pt_fn_measurements")
            .find("Sequential_source")
            .findall("elem") )
            

        seqsrc_type = (
            .find("SequentialProp_record_info")
            .find("SequentialProp")
            .find("SeqSource")
            .find("SeqSource")
            .find("SeqSourceType")
            .text
        )

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

        forward_mass = [
            float(x.find("ForwardProp").find("FermionAction").find("Kappa").text)
            for x in forwar_props
        ]
        current_mass_in = (
            root.find("bar3ptfn")
            .find("Propagator_record_info")
            .find("Propagator")
            .find("ForwardProp")
            .find("FermionAction")
            .find("Kappa")
            .text
        )

        current_mass_out = 

        head, record = cf.ReadRecord(file_in)

        while head != b"":
            if not head[16:].startswith(b"meta-xml"):
                raise IOError("Expecting meta-xml record")
            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

            has_third = not (
                root.find("Forward_prop_headers").find("Third_forward_prop") == None
            )

            baryon_number = int(root.find("baryon-number").text)

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

            if κ1 == κ2:
                ferm_act_string += "_nf0"
            elif has_third:
                ferm_act_string += "_nf3"
            else:
                ferm_act_string += "_nf2"

            # Chroma throws an error if both props have different smearing, need
            # only check one each at source and sink

            source_string = (
                root.find("SourceSinkType").find("source_type_1").text.lower()
            )
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
                    f"{SourceNames(source_string)}"
                    + f"_{SmearingNames(source_kind)}"
                    + f"_{source_param}_{source_intparam}"
                )
            else:
                source_string = SourceNames(source_string)
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
                    f"{SinkNames(sink_string)}"
                    + f"_{SmearingNames(sink_kind)}"
                    + f"_{sink_param}_{sink_intparam}"
                )
            else:
                sink_string = SinkNames(sink_string)
            source_sink_string = f"{source_string}-{sink_string}"

            head, record = cf.ReadRecord(file_in)

            if not head[16:].startswith(b"baryons-bin"):
                raise IOError("Expecting baryons-bin record")

            record = np.frombuffer(record, ">f8").reshape(
                baryon_number, num_mom, latt_size[3], 2
            )

            for n, p in enumerate(mom_list):
                p_str = "p" + "".join([f"{p_i:+d}" for p_i in p])

                for b in range(baryon_number):
                    bar_str = BaryonNames(b)

                    record_sliced = record[b, n]

                    if (
                        type(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][bar_str]
                        )
                        == collections.defaultdict
                        and len(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][bar_str].keys()
                        )
                        == 0
                    ):
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            p_str
                        ][bar_str] = [record_sliced]
                    else:
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            p_str
                        ][bar_str].append(record_sliced)

            head, record = cf.ReadRecord(file_in)

            if time_rev:
                if not head[16:].startswith(b"baryons-trev-bin"):
                    raise IOError("Expecting baryons-trev-bin record")

                record = np.frombuffer(record, ">f8").reshape(
                    baryon_number, num_mom, latt_size[3], 2
                )

                for n, p in enumerate(mom_list):
                    p_str = "p" + "".join([f"{p_i:+d}" for p_i in p])

                    for b in range(baryon_number):
                        bar_str = BaryonNames(b)

                        record_sliced = record[b, n]

                        if (
                            type(
                                data_trev[latt_size_str][ferm_act_string][κ_str][
                                    source_sink_string
                                ][p_str][bar_str]
                            )
                            == collections.defaultdict
                            and len(
                                data_trev[latt_size_str][ferm_act_string][κ_str][
                                    source_sink_string
                                ][p_str][bar_str].keys()
                            )
                            == 0
                        ):
                            data_trev[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][bar_str] = [record_sliced]
                        else:
                            data_trev[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][bar_str].append(record_sliced)

                head, record = cf.ReadRecord(file_in)

    for latt_size, lvl1 in data.items():
        for ferm_act, lvl2 in lvl1.items():
            for κ, lvl3 in lvl2.items():
                for source_sink, lvl4 in lvl3.items():
                    for p, lvl5 in lvl4.items():
                        out_dir = (
                            loc
                            + f"/barspec/{latt_size}/{ferm_act}/{κ}/"
                            + f"/{source_sink}/{p}/"
                        )

                        os.system(f"mkdir -p {out_dir}")

                        for b, lvl6 in lvl5.items():
                            ncfg = len(lvl6)
                            out_name = f"barspec_{b}_{ncfg}cfgs.pickle"
                            with open(out_dir + out_name, "wb") as file_out:
                                pickle.dump(np.array(lvl6), file_out)

    if time_rev:
        for latt_size, lvl1 in data_trev.items():
            for ferm_act, lvl2 in lvl1.items():
                for κ, lvl3 in lvl2.items():
                    for source_sink, lvl4 in lvl3.items():
                        for p, lvl5 in lvl4.items():
                            out_dir = (
                                loc
                                + f"/barspec/{latt_size}/{ferm_act}/{κ}/"
                                + f"/{source_sink}/{p}/"
                            )

                            os.system(f"mkdir -p {out_dir}")

                            for b, lvl6 in lvl5.items():
                                ncfg = len(lvl6)
                                out_name = f"barspec_{b}_timerev_{ncfg}cfgs.pickle"
                                with open(out_dir + out_name, "wb") as file_out:
                                    pickle.dump(np.array(lvl6), file_out)

    return


def rec_dd():
    return collections.defaultdict(rec_dd)


def SourceNames(name):
    names = {"shell_source": "sh", "point_source": "pt"}
    return names[name]


def SinkNames(name):
    names = {"shell_sink": "sh", "point_sink": "pt"}
    return names[name]


def SmearingNames(name):
    names = {"gauge_inv_jacobi": "gij"}
    return names[name]


def BaryonNames(number):
    """I'm using a dict for readability, yes a list would work just as well"""
    names = {
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
    return names[number]
