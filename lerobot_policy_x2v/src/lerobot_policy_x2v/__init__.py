# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from lerobot_policy_x2v.configuration_x2v import X2VConfig
from lerobot_policy_x2v.modeling_x2v import X2VPolicy  
from lerobot_policy_x2v.processor_x2v import make_x2v_diffusion_pre_post_processors

__all__ = [
    "X2VConfig",
    "X2VPolicy",
    "make_x2v_diffusion_pre_post_processors",
]