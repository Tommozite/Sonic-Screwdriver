from .bar3ptfn import read_bar3ptfn, write_bar3ptfn, emergency_write_bar3ptfn


class bar3ptfn:
    def __init__(self):
        self.attribute_names = [
            "lattice_size",
            "fermion_action",
            "kappa",
            "source_sink",
            "sequential_source",
            "momentum",
            "form_factor",
        ]
        self.reader = read_bar3ptfn
        self.writer = write_bar3ptfn
        self.emergency_dump = emergency_write_bar3ptfn
