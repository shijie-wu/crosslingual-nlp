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
    pawsx = "pawsx"
    mldoc = "mldoc"
    langid = "langid"
    conllner = "ner-conll"
    wikiner = "ner-wiki"
    udpos = "udpos"
    parsing = "parsing"


class Schedule(NamedEnum):
    linear = "linear"
    invsqroot = "invsqroot"
    reduceOnPlateau = "reduceOnPlateau"
