# This code is taken from the OneIE v0.4.7 preprocessing/process_ace.py script.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.
    Args:
        text (str): text to unescape.

    Returns:
        str: unescaped string.
    """
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace("\n", " ")

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            dict: a dict of instance variables.
        """
        return {"text": recover_escape(self.text), "start": self.start, "end": self.end}


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            "text": recover_escape(self.text),
            "entity_id": self.entity_id,
            "mention_id": self.mention_id,
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "entity_subtype": self.entity_subtype,
            "mention_type": self.mention_type,
        }
        if self.value:
            entity_dict["value"] = self.value
        return entity_dict


@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "mention_id": self.mention_id,
            "role": self.role,
            "text": recover_escape(self.text),
        }


@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "relation_subtype": self.relation_subtype,
            "arg1": self.arg1.to_dict(),
            "arg2": self.arg2.to_dict(),
        }


@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "mention_id": self.mention_id,
            "role": self.role,
            "text": recover_escape(self.text),
        }


@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "event_id": self.event_id,
            "mention_id": self.mention_id,
            "event_type": self.event_type,
            "event_subtype": self.event_subtype,
            "trigger": self.trigger.to_dict(),
            "arguments": [arg.to_dict() for arg in self.arguments],
        }


@dataclass
class Sentence(Span):
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "sent_id": self.sent_id,
            "tokens": [recover_escape(t) for t in self.tokens],
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "events": [event.to_dict() for event in self.events],
            "start": self.start,
            "end": self.end,
            "text": recover_escape(self.text).replace("\t", " "),
        }


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            "doc_id": self.doc_id,
            "sentences": [sent.to_dict() for sent in self.sentences],
        }
