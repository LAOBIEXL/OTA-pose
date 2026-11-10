"""Training and evaluation functions for OTA-Pose."""
from utils.utils_registry import Registry

CORE_FUNCTION_REGISTRY = Registry("CORE_FUNCTION")

from .base import BaseFunction, AverageMeter, build_core_function
from .function_no_supp_targ_5_frame_ot import Function_No_Supp_Targ_5_Frame_OT

__all__ = [
    "CORE_FUNCTION_REGISTRY", "BaseFunction", "AverageMeter", "build_core_function",
    "Function_No_Supp_Targ_5_Frame_OT",
]
