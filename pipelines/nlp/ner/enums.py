from ..enums import ExplicitEnum, EqualEnum


class TaggingScheme(ExplicitEnum, EqualEnum):
    IO = "io" # Inside-Outside
    BIO = "bio" # Beginning-Inside-Outside
    BIEO = "bieo" # Beginning-Inside-End-Outside