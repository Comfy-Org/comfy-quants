"""Process-local component registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from comfy_quants.core.errors import AdapterNotFoundError


@dataclass
class ComfyQuantsRegistry:
    """In-process registry for built-in and extension components."""

    _adapters: dict[str, Any] = field(default_factory=dict)
    _algorithms: dict[str, Any] = field(default_factory=dict)
    _formats: dict[str, Any] = field(default_factory=dict)
    _backends: dict[str, Any] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def register_adapter(self, adapter: Any, *, name: str | None = None, replace: bool = True) -> None:
        key = name or getattr(adapter, "family")
        with self._lock:
            if not replace and key in self._adapters:
                raise KeyError(f"adapter already registered: {key}")
            self._adapters[key] = adapter

    def get_adapter(self, family: str) -> Any:
        with self._lock:
            try:
                return self._adapters[family]
            except KeyError as exc:
                raise AdapterNotFoundError(f"unknown model family {family!r}; available: {self.list_adapters()}") from exc

    def list_adapters(self) -> list[str]:
        with self._lock:
            return sorted(self._adapters)

    def register_algorithm(self, algorithm: Any, *, name: str | None = None, replace: bool = True) -> None:
        key = name or getattr(algorithm, "name")
        with self._lock:
            if not replace and key in self._algorithms:
                raise KeyError(f"algorithm already registered: {key}")
            self._algorithms[key] = algorithm

    def get_algorithm(self, name: str) -> Any:
        with self._lock:
            if name not in self._algorithms:
                raise KeyError(f"unknown algorithm {name!r}; available: {self.list_algorithms()}")
            return self._algorithms[name]

    def list_algorithms(self) -> list[str]:
        with self._lock:
            return sorted(self._algorithms)

    def register_format(self, quant_format: Any, *, name: str | None = None, replace: bool = True) -> None:
        key = name or getattr(quant_format, "name")
        with self._lock:
            if not replace and key in self._formats:
                raise KeyError(f"format already registered: {key}")
            self._formats[key] = quant_format

    def get_format(self, name: str) -> Any:
        with self._lock:
            if name not in self._formats:
                raise KeyError(f"unknown format {name!r}; available: {self.list_formats()}")
            return self._formats[name]

    def list_formats(self) -> list[str]:
        with self._lock:
            return sorted(self._formats)

    def register_backend(self, backend: Any, *, name: str | None = None, replace: bool = True) -> None:
        key = name or getattr(backend, "backend_name")
        with self._lock:
            if not replace and key in self._backends:
                raise KeyError(f"backend already registered: {key}")
            self._backends[key] = backend

    def get_backend(self, name: str) -> Any:
        with self._lock:
            if name not in self._backends:
                raise KeyError(f"unknown backend {name!r}; available: {self.list_backends()}")
            return self._backends[name]

    def list_backends(self) -> list[str]:
        with self._lock:
            return sorted(self._backends)


registry = ComfyQuantsRegistry()
