from .messpec import read_messpec, write_messpec, emergency_write_messpec


class messpec:
    def __init__(self):
        self.attribute_names = [
            "lattice_size",
            "fermion_action",
            "kappa",
            "source_sink",
            "momentum",
            "gamma",
        ]
        self.reader = read_messpec
        self.writer = write_messpec
        self.emergency_dump = emergency_write_messpec
