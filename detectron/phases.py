from enum import Enum


class OpPhase(Enum):
    """
    Enum represents the various phases that were available in the annotated data
    """

    FIRST_INCISION = 1
    PEDICLE_PACKAGE = 2
    VASCULAR_DISSECTION = 3
    MESOCOLON_GEROTA = 4
    LATERAL_MOBILISATION = 5
    TME = 6
