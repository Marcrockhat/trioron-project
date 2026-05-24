"""Core substrate primitives.  See spec §2."""
from .state import CellState
from .roles import CellRole
from .epigenome import (
    LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE,
    PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, RECYCLABLE,
    WEIGHT_TIED_LINEAGE, EXPRESSION_GENES,
    has_gene, set_gene, clear_gene, primary_phenotype,
)
from .envelope import Envelope
from .arena import Arena
from .cell import Cell
from .lineage import Lineage
from .graph import CellGraph
from .scheduler import Scheduler, Bucket, DispatchPlan, ForwardFn
from .construct import Substrate, Base, construct
