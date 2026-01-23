"""lerobot_policy_inc package initialization."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use lerobot_policy_inc."
    )

from lerobot_policy_inc.src.lerobot_policy_inc.configuration_inc import INCConfig
from lerobot_policy_inc.src.lerobot_policy_inc.modeling_inc import INCPolicy
from lerobot_policy_inc.src.lerobot_policy_inc.processor_inc import make_inc_pre_post_processors

__all__ = [
    "INCConfig",
    "INCPolicy",
    "make_inc_pre_post_processors",
]
