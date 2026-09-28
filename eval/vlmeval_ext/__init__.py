"""
egotools eval extensions to VLMEvalKit.

This package adds the EgotoolsBench dataset adapter without modifying
upstream VLMEvalKit code. Calling :func:`vlmeval_ext.register.register_egotools_bench` injects
:class:`EgotoolsBench` into VLMEvalKit's dataset registry so that
``vlmeval.dataset.build_dataset(name)`` can construct it.
"""

from .egotools_dataset import EGOTOOLS_VERSIONS, EgotoolsBench

__all__ = ["EgotoolsBench", "EGOTOOLS_VERSIONS"]
