"""1-Lipschitz sphere-tracing package."""
from .model import FTheta, ConvexPotentialLayer, MaxMin, GroupSort
from .sphere_tracing import trace_unrolled, trace_nograd
from .config import SCENE, BLENDER_SCENE, OUT_DIR, ModelConfig, TraceConfig, TrainConfig
