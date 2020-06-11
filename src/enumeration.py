from dataclasses import asdict, dataclass


@dataclass
class Enum:
    def choices(self):
        return list(asdict(self).values())


@dataclass
class Split(Enum):
    train: str = "train"
    dev: str = "dev"
    test: str = "test"


@dataclass
class Task(Enum):
    xnli: str = "xnli"
    pawsx: str = "pawsx"
    mldoc: str = "mldoc"
    langid: str = "langid"
    conllner: str = "ner-conll"
    wikiner: str = "ner-wiki"
    udpos: str = "udpos"
    parsing: str = "parsing"


@dataclass
class Schedule(Enum):
    linear: str = "linear"
    invsqroot: str = "invsqroot"
    reduceOnPlateau: str = "reduceOnPlateau"
