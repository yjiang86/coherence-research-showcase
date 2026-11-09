"""Self-contained CPU coherence kernel quadrant."""

from .api import CoherenceResult, compute_single, iter_multi_rhs
from .kernel import kernel_owned_array_payload_upper_bound_bytes
from .policy import ConvolutionPolicy

__all__ = (
    "CoherenceResult",
    "ConvolutionPolicy",
    "compute_single",
    "iter_multi_rhs",
    "kernel_owned_array_payload_upper_bound_bytes",
)
