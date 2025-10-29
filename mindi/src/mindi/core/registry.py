from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Type, TypeVar

from .client import InferenceClient
from .dataset import DatasetProvider

T = TypeVar("T")

class RegistryBase(Generic[T]):
    """Registry helper"""

    @classmethod
    def _entries(cls) -> Dict[str, T]:
        storage = cls.__dict__.get("_registry_entries")
        if storage is None:
            storage = {}
            setattr(cls, "_registry_entries", storage)
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

