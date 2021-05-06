# %%
import gc
import numpy as np
import xml.etree.ElementTree as ET
from . import core_functions as cf
import os
import itertools
import dill as pickle
import collections
import sys
from psutil import virtual_memory

magic_bytes = b"Eg\x89\xab"
Nd = 4


def unpack_messpec(filelist_iter, loc=""):
    data = rec_dd()
    file_count = 0

    emergency_dumps = 0
    max_memory = virtual_memory().total / 100
    file_size = os.path.getsize(filelist_iter[0])
    check_interval = max_memory // file_size

    print("reading limes")
    for filename in filelist_iter:
        if (file_count := file_count + 1) % check_interval == 0:
            print(f"reading lime {file_count}")
            if get_obj_size(data) > max_memory:
                print("Emergency dumping data")
                emergency_dumps += 1
                for latt_size, lvl1 in data.items():
                    for ferm_act, lvl2 in lvl1.items():
                        for κ, lvl3 in lvl2.items():
                            for source_sink, lvl4 in lvl3.items():
                                for p, lvl5 in lvl4.items():
                                    out_dir = (
                                        loc
                                        + f"/messpec/{latt_size}/{ferm_act}/{κ}/"
                                        + f"/{source_sink}/{p}/"
                                    )

                                    os.system(f"mkdir -p {out_dir}")

                                    for γ, lvl6 in lvl5.items():
                                        out_name = (
                                            f"messpec_{γ}.pickle.temp{emergency_dumps}"
                                        )
                                        with open(out_dir + out_name, "wb") as file_out:
                                            pickle.dump(np.array(lvl6), file_out)
                data = rec_dd()

        file_in = open(filename.strip(), "rb")
        head, record = cf.ReadRecord(file_in)
        if head[:4] != magic_bytes:
            raise IOError("Record header missing magic bytes.")

        if not head[16:].startswith(b"qcdsfDir"):
            raise IOError("Missing qcdsfDir record")

        tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
        root = tree.getroot()
        latt_size = [
            int(s)
            for s in root.find("ProgramInfo")
            .find("Setgeom")
            .find("latt_size")
            .text.split(" ")
        ]
        seen = set()
        latt_size_str = "x".join(
            str(x) for x in latt_size if not (x in seen or seen.add(x))
        )
        mom2_max = int(root.find("Input").find("Param").find("mom2_max").text)
        num_mom, mom_list = cf.CountMom(mom2_max, Nd)

        head, record = cf.ReadRecord(file_in)

        while head != b"":
            if not head[16:].startswith(b"meta-xml"):
                raise IOError("Expecting meta-xml record")
            tree = ET.ElementTree(ET.fromstring(record.decode("utf-8", "ignore")))
            root = tree.getroot()

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
            if not head[16:].startswith(b"mesons-bin"):
                raise IOError("Expecting mesons-bin record")

            record = np.frombuffer(record, ">f8").reshape(
                16, 16, num_mom, latt_size[3], 2
            )

            for n, p in enumerate(mom_list):
                p_str = "p" + "".join([f"{p_i:+d}" for p_i in p])

                for γ1, γ2 in itertools.product(range(16), range(16)):

                    γ_str = f"{γString(γ1)}-{γString(γ2)}"
                    record_sliced = record[γ1, γ2, n]
                    if (
                        type(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][γ_str]
                        )
                        == collections.defaultdict
                        and len(
                            data[latt_size_str][ferm_act_string][κ_str][
                                source_sink_string
                            ][p_str][γ_str].keys()
                        )
                        == 0
                    ):
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            p_str
                        ][γ_str] = [record_sliced]
                    else:
                        data[latt_size_str][ferm_act_string][κ_str][source_sink_string][
                            p_str
                        ][γ_str].append(record_sliced)

            head, record = cf.ReadRecord(file_in)

    print("writing pickles")
    if emergency_dumps > 0:
        for latt_size, lvl1 in data.items():
            for ferm_act, lvl2 in lvl1.items():
                for κ, lvl3 in lvl2.items():
                    for source_sink, lvl4 in lvl3.items():
                        for p, lvl5 in lvl4.items():
                            out_dir = (
                                loc
                                + f"/messpec/{latt_size}/{ferm_act}/{κ}/"
                                + f"/{source_sink}/{p}/"
                            )

                            os.system(f"mkdir -p {out_dir}")

                            for γ, lvl6 in lvl5.items():
                                temp_name = f"messpec_{γ}.pickle"
                                out_data = []
                                for ed in range(emergency_dumps):
                                    with open(
                                        out_dir + temp_name + f".temp{ed+1}", "rb"
                                    ) as file_in:
                                        out_data.append(pickle.load(file_in))
                                    os.remove(out_dir + temp_name + f".temp{ed+1}")
                                out_data.append(np.array(lvl6))
                                out_data = np.concatenate(out_data, axis=0)
                                ncfg = len(out_data)
                                out_name = f"messpec_{γ}_{ncfg}cfgs.pickle"
                                with open(out_dir + out_name, "wb") as file_out:
                                    pickle.dump(out_data, file_out)

    else:
        for latt_size, lvl1 in data.items():
            for ferm_act, lvl2 in lvl1.items():
                for κ, lvl3 in lvl2.items():
                    for source_sink, lvl4 in lvl3.items():
                        for p, lvl5 in lvl4.items():
                            out_dir = (
                                loc
                                + f"/messpec/{latt_size}/{ferm_act}/{κ}/"
                                + f"/{source_sink}/{p}/"
                            )

                            os.system(f"mkdir -p {out_dir}")

                            for γ, lvl6 in lvl5.items():
                                ncfg = len(lvl6)
                                out_name = f"messpec_{γ}_{ncfg}cfgs.pickle"
                                with open(out_dir + out_name, "wb") as file_out:
                                    pickle.dump(np.array(lvl6), file_out)

    return


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


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {
            o_id: o
            for o_id, o in all_refr
            if o_id not in marked and not isinstance(o, type)
        }

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
