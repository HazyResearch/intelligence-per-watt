from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis.base import AnalysisProvider
    from ..clients.base import InferenceClient
    from ..datasets.base import DatasetProvider
    from ..visualization.base import VisualizationProvider

T = TypeVar("T")


class RegistryBase(Generic[T]):
    """Registry helper"""

    @classmethod
    def _entries(cls) -> Dict[str, T]:
        # Use class-specific attribute name to ensure isolation between registry types
        attr_name = f"_registry_entries_{cls.__name__}"
        storage = getattr(cls, attr_name, None)
        if storage is None:
            storage = {}
            setattr(cls, attr_name, storage)
        return storage  # type: ignore[return-value]

    @classmethod
    def register(cls, key: str) -> Callable[[T], T]:
        def decorator(entry: T) -> T:
            entries = cls._entries()
            if key in entries:
                raise ValueError(f"{cls.__name__} already has an entry for '{key}'")
            entries[key] = entry
            return entry

        return decorator

    @classmethod
    def register_value(cls, key: str, value: T) -> T:
        entries = cls._entries()
        if key in entries:
            raise ValueError(f"{cls.__name__} already has an entry for '{key}'")
        entries[key] = value
        return value

    @classmethod
    def get(cls, key: str) -> T:
        try:
            return cls._entries()[key]
        except KeyError as exc:
            raise KeyError(f"{cls.__name__} does not have an entry for '{key}'") from exc

    @classmethod
    def create(cls, key: str, *args: Any, **kwargs: Any) -> Any:
        entry = cls.get(key)
        if not callable(entry):
            raise TypeError(
                f"{cls.__name__} entry '{key}' is not callable and cannot be instantiated"
            )
        return entry(*args, **kwargs)

    @classmethod
    def items(cls):
        return tuple(cls._entries().items())

    @classmethod
    def clear(cls) -> None:
        cls._entries().clear()


class ClientRegistry(RegistryBase[Type["InferenceClient"]]):
    """Registry for inference clients."""


class DatasetRegistry(RegistryBase[Type["DatasetProvider"]]):
    """Registry for dataset providers."""


class AnalysisRegistry(RegistryBase[Type["AnalysisProvider"]]):
    """Registry for analysis providers."""


class VisualizationRegistry(RegistryBase[Type["VisualizationProvider"]]):
    """Registry for visualization providers."""
