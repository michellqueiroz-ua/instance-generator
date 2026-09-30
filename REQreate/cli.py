"""Command line entry point: ``reqreate``.

The point of this is that someone can ``pip install reqreate`` and then run
``reqreate app`` to get the web interface in their browser, with everything
executing on their own machine. That avoids depending on a hosted deployment,
and it means OpenStreetMap downloads leave from their own IP rather than an
address shared with every other user of a hosting platform -- which is what
gets requests refused by the public Overpass API.
"""

import argparse
import json
import os
import sys

from . import __version__

APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "app.py")


def _run_app(args):
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "Streamlit is not installed. Reinstall with the web interface:\n"
            "    pip install 'reqreate[app]'"
        )

    # Streamlit is driven through its own CLI, which reads sys.argv.
    argv = [
        "streamlit", "run", APP_PATH,
        "--server.port", str(args.port),
        "--server.address", args.address,
    ]
    if args.no_browser:
        argv += ["--server.headless", "true"]
    sys.argv = argv
    sys.exit(stcli.main())


def _run_generate(args):
    from .input_json import input_json

    config_path = os.path.abspath(args.config)
    if os.path.isdir(config_path):
        configs = sorted(
            os.path.join(config_path, filename)
            for filename in os.listdir(config_path)
            if filename.endswith(".json") and not filename.startswith("._")
        )
    else:
        configs = [config_path]

    succeeded = 0
    failed = []
    skipped = 0
    for config in configs:
        try:
            if args.skip_existing:
                with open(config, encoding="utf-8") as config_file:
                    data = json.load(config_file)
                network = data.get("network")
                if isinstance(network, str):
                    output_name = os.path.basename(config).replace(".json", "") + "_1.csv"
                    output_path = os.path.join(
                        os.getcwd(), network, "csv_format", args.output or "", output_name
                    )
                    if os.path.isfile(output_path):
                        skipped += 1
                        continue

            directory = os.path.dirname(config)
            input_json(directory + os.sep, os.path.basename(config), args.output or "")
            succeeded += 1
        except Exception as error:
            if not args.continue_on_error:
                raise
            failed.append(config)
            print(f"Failed to generate {config}: {error}", file=sys.stderr)

    if args.continue_on_error:
        print(
            f"Generation summary: {succeeded} succeeded, {len(failed)} failed, "
            f"{skipped} skipped."
        )
        if failed:
            print("Failed configs:")
            for config in failed:
                print(f"  {config}")
            return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="reqreate",
        description="Generate on-demand transportation instances from real OpenStreetMap networks.",
    )
    parser.add_argument("--version", action="version", version=f"reqreate {__version__}")
    sub = parser.add_subparsers(dest="command")

    app = sub.add_parser("app", help="open the web interface in a browser (runs locally)")
    app.add_argument("--port", type=int, default=8501, help="port to serve on (default: 8501)")
    app.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    app.add_argument(
        "--address", default="localhost",
        help="interface to bind to (default: localhost; use 0.0.0.0 to serve on a network)",
    )
    app.set_defaults(func=_run_app)

    gen = sub.add_parser("generate", help="generate instances from JSON configuration files")
    gen.add_argument("config", help="path to a JSON configuration file or a folder of files")
    gen.add_argument("-o", "--output", help="folder name to save the instance under")
    gen.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip configurations whose output already exists",
    )
    gen.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue after a configuration fails and summarize results",
    )
    gen.set_defaults(func=_run_generate)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # Bare `reqreate` is most likely someone wanting the interface.
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
