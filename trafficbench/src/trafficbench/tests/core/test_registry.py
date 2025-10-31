from __future__ import annotations

import pytest

from trafficbench.core.registry import ClientRegistry, DatasetRegistry, RegistryBase


class DummyEntry:
    def __init__(self, value: str) -> None:
        self.value = value


class DummyRegistry(RegistryBase[type[DummyEntry] | DummyEntry]):
    pass


def test_register_and_get() -> None:
    DummyRegistry.clear()

    @DummyRegistry.register("foo")
    class Foo(DummyEntry):
        pass

    retrieved = DummyRegistry.get("foo")
    assert retrieved is Foo


def test_register_duplicate_raises() -> None:
    DummyRegistry.clear()

    @DummyRegistry.register("dup")
    class Dup(DummyEntry):
        pass

    with pytest.raises(ValueError):
        DummyRegistry.register("dup")(DummyEntry)


def test_create_instantiates_registered_class() -> None:
    DummyRegistry.clear()

    @DummyRegistry.register("inst")
    class Inst(DummyEntry):
        def __init__(self, value: str) -> None:
            super().__init__(value)

    instance = DummyRegistry.create("inst", "value")
    assert isinstance(instance, Inst)
    assert instance.value == "value"


def test_create_non_callable_entry_raises() -> None:
    DummyRegistry.clear()
    DummyRegistry.register_value("value", DummyEntry("constant"))

    with pytest.raises(TypeError):
        DummyRegistry.create("value")


def test_client_registry_independent_from_dataset_registry() -> None:
    ClientRegistry.clear()
    DatasetRegistry.clear()

    @ClientRegistry.register("client")
    class Client:
        pass

    @DatasetRegistry.register("dataset")
    class Dataset:
        pass

    assert ClientRegistry.get("client") is Client
    assert DatasetRegistry.get("dataset") is Dataset
