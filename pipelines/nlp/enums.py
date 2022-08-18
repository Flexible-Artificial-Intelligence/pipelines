from enum import Enum


class ExplicitEnum(Enum):
    @classmethod
    def _missing_(cls, value: str) -> None:
        keys = list(cls._value2member_map_.keys())
        raise ValueError(f"`{value}` is not a valid or not supported `{cls.__name__}`, select one of `{keys}`.")


class EqualEnum(Enum):
    def __eq__(self, other: Enum) -> bool:
        return self.value == other.value


class PaddingSide(ExplicitEnum, EqualEnum):
    RIGHT = "right"
    LEFT = "left"