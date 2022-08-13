from pickled_lime.unpack_limes import unpack_limes

# an iterable of all file paths to be read
file_list = [
    f"/media/tomas/tardis/chroma_2pt/24x48/b6p00kp133300kp133300/"
    + f"twopoint/messpec_qcdsf.0.b6p00kp133300kp133300-24x48.{i}.lime"
    for i in range(1, 1001)
]

# out root location
out_loc = f"pickles"

# filter dict, will only create files of the given values.
# pass no dict to get all data from lime files
filter = {"momentum": "p+0+0+0", "gamma": "g5-g5"}

# same function for barspec, messpec and bar3ptfn, first argument string sets measure
unpack_limes("messpec", file_list, loc=out_loc, filter_dict=filter)
