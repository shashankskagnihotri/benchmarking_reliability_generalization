"""The ``cospgd.functions`` object used by ``EncoderDecoder``.

``mmseg.models.segmentors.encoder_decoder`` imports this name. The methods
are the cospgd 0.1.0 projection steps kept in ``vlseg.cospgd_functions``.
"""

from semsegbench.vlseg.cospgd_functions import functions

__all__ = ["functions"]
