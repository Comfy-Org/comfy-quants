"""Project-specific exceptions."""


class ComfyQuantsError(Exception):
    """Base class for all Comfy Quants exceptions."""


class ConfigurationError(ComfyQuantsError):
    """Raised when a configuration file or CLI argument is invalid."""


class AdapterNotFoundError(ComfyQuantsError):
    """Raised when no model adapter is registered for a requested family."""


class ManifestError(ComfyQuantsError):
    """Raised when an artifact or job manifest is invalid."""


class PayloadWriteError(ComfyQuantsError):
    """Raised when artifact tensor payload writing fails."""


class CompatibilityError(ComfyQuantsError):
    """Raised when an artifact does not meet a requested compatibility level."""
