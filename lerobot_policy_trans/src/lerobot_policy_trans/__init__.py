"""lerobot_policy_ditflow package initialization."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_ditflow."
    )

from lerobot_policy_trans.configuration_trans import TransDiffusionConfig
from lerobot_policy_trans.modeling_trans import TransDiffusionPolicy
from lerobot_policy_trans.processor_trans import make_trans_pre_post_processors
   

__all__ = [
    "TransDiffusionConfig",
    "TransDiffusionPolicy",
    "make_trans_pre_post_processors",
]
