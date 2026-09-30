"""Validate REQreate configuration files against the bundled JSON Schema."""

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator


_SCHEMA = json.loads(
    files("REQreate").joinpath("schema/config.schema.json").read_text(encoding="utf-8")
)
_VALIDATOR = Draft202012Validator(_SCHEMA)


def _format_path(parts):
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def validate_config(path_or_dict):
    """Return readable validation errors for a JSON path or configuration dict."""
    if isinstance(path_or_dict, (str, Path)):
        path = Path(path_or_dict)
        try:
            with path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [f"Could not read {path}: {exc}"]
    else:
        config = path_or_dict

    errors = sorted(
        _VALIDATOR.iter_errors(config),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    return [
        f"{_format_path(error.absolute_path)}: {error.message}"
        for error in errors
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a REQreate JSON configuration")
    args = parser.parse_args(argv)

    errors = validate_config(args.config)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"{args.config}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
