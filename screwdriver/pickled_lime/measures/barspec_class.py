from .barspec import read_barspec, write_barspec, emergency_write_barspec


class barspec:
    def __init__(self):
        self.attribute_names = [
            "lattice_size",
            "fermion_action",
            "kappa",
            "source_sink",
            "momentum",
            "baryon",
        ]
        self.reader = read_barspec
        self.writer = write_barspec
        self.emergency_dump = emergency_write_barspec
