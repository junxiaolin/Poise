"""Portable Docker command construction; no shell or hardware access here."""

from pathlib import PurePosixPath


def sdk_command(container: str, sdk_root: str, python: str) -> list[str]:
    for name, value in (("sdk_container", container), ("sdk_root", sdk_root), ("sdk_python", python)):
        if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
            raise ValueError(f"{name} must be a nonempty single-line string")
    if container.startswith("-") or python.startswith("-"):
        raise ValueError("Container and Python executable cannot start with '-'")
    root = PurePosixPath(sdk_root)
    if not root.is_absolute() or ":" in sdk_root or ".." in root.parts:
        raise ValueError("sdk_root must be an absolute container path without ':' or '..'")
    # Each environment setting is one argv item, including paths with spaces.
    return ["docker", "exec", "-i", "-e", f"SHARPA_SDK_ROOT={root}",
            "-e", f"LD_LIBRARY_PATH={root / 'lib'}"]
