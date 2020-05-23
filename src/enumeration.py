from enum import Enum


class NamedEnum(Enum):
    def __str__(self):
        return self.value


class Split(NamedEnum):
    train = "train"
    dev = "dev"
    test = "test"


class Task(NamedEnum):
    xnli = "xnli"
    mldoc = "mldoc"
    langid = "langid"
    conllner = "conllner"
    wikiner = "wikiner"
    udpos = "udpos"
    parsing = "parsing"


class Schedule(NamedEnum):
    linear = "linear"
    invsqroot = "invsqroot"
    reduceOnPlateau = "reduceOnPlateau"
