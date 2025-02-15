from enum import Enum


class Labels(str, Enum):
    """
    Task labels.
    """

    IMAGE_GENERATED = "image_generated"
    TEXT_GENERATED = "text_generated"
    FULLY_HUMAN = "fully_human"
    FULLY_GENERATED = "fully_generated"
