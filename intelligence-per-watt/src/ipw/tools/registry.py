"""ToolRegistry — registers tool classes (not instances) keyed by tool name.

Follows the RegistryBase pattern shared by AgentRegistry, DatasetRegistry,
ClientRegistry, etc. (see `ipw.core.registry`). Adds build_tool_descriptions()
for prompt construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from ipw.core.registry import RegistryBase
from ipw.tools.base import BaseTool


class ToolRegistry(RegistryBase[Type[BaseTool]]):
    """Registry of tool classes (keyed by tool name string)."""

    @classmethod
    def build_tool_descriptions(cls, tool_ids: List[str]) -> List[Dict[str, Any]]:
        """Return function-calling-style schemas for the named tools.

        Unknown tool ids are skipped (not raised); a partial result is the right
        behavior when an agent's tool list straddles registered and pending tools.
        """
        descs: List[Dict[str, Any]] = []
        for name in tool_ids:
            if not cls.has(name):
                continue
            tool_cls = cls.get(name)
            spec = tool_cls.spec
            descs.append({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            })
        return descs


__all__ = ["ToolRegistry"]
