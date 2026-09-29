"""The world the agent's tools act on during an eval: state, mocked tools, and faults.

An item can describe its environment:

  "environment": {
    "state": {"orders": {"55": {"status": "delivered", "refunded": false}}},
    "mocks": {"weather": {"returns": {"temp_c": 18}}},
    "faults": [{"tool": "refund", "on_call": 1, "effect": "error", "message": "payments down"}],
    "seed": 7
  }
  "expected": {"state": {"orders": {"55": {"refunded": true}}}}

The agent gets the environment through `context.env` and calls its tools
through it (`context.env.call("refund", order="55")`), so every call is
recorded, faults can be injected, and the final state can be checked. Real
tool implementations are registered with `@tool` and change `env.state`;
mocked tools just return a fixed value.
"""

from warden_sdk.evals.environment.env import (
    Environment,
    ToolError,
    check_environment,
    tool,
)
from warden_sdk.evals.environment.faults import Fault, with_faults

__all__ = ["Environment", "Fault", "ToolError", "check_environment", "tool", "with_faults"]
