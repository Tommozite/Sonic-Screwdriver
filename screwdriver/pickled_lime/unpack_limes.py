import importlib

from . import core_functions as cf
from . import readers


magic_bytes = cf.magic_bytes
max_memory = 200000000  # 200MB hard coded for now


def filter_data(data, filter_dict, attribute_names):
    result = {
        key: value
        for key, value in data.items()
        if readers.read_filter_dict(filter_dict, attribute_names, key)
    }
    return result


def emergency_dumps(data, loc, emergency_dump_count, emergency_writer):
    if cf.get_obj_size(data) > max_memory:
        print("Emergency dumping data")
        emergency_dump_count += 1
        emergency_writer(data, loc, emergency_dump_count)

        return {}
    else:
        return data


def unpack_limes(
    measure_type,
    filelist_iter,
    loc=".",
    filter_dict=None,
    transition_form=False,
    simplify=False,
):
    measure = getattr(
        importlib.import_module(
            ".measures." + measure_type + "_class", package="screwdriver.pickled_lime"
        ),
        measure_type,
    )()
    data_out = {}
    emergency_dump_count = 0
    print("Reading Limes")
    for file_count, filename in enumerate(filelist_iter):
        if (file_count + 1) % 10 == 0:
            print(f"Lime file {file_count+1}")

        data_out = emergency_dumps(
            data_out, loc, emergency_dump_count, measure.emergency_dump
        )

        file = open(filename, "rb")

        data = measure.reader(file, simplify, transition_form)
        data_filtered = filter_data(data, filter_dict, measure.attribute_names)
        for key, value in data_filtered.items():
            if key in data_out:
                data_out[key].append(value)
            else:
                data_out[key] = [value]

    measure.writer(data_out, loc, emergency_dump_count)

