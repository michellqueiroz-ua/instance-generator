from datetime import datetime
from pathlib import Path


def list_networks(working_dir="."):
    """Return built network folders with their modification time and size."""
    networks = []
    for folder in Path(working_dir).iterdir():
        if not folder.is_dir():
            continue
        network_file = folder / "pickle" / f"{folder.name}.network.class.pkl"
        if not network_file.is_file():
            continue
        size_bytes = sum(
            file.stat().st_size for file in folder.rglob("*") if file.is_file()
        )
        networks.append({
            "name": folder.name,
            "modified": datetime.fromtimestamp(
                network_file.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "size_bytes": size_bytes,
        })
    return sorted(networks, key=lambda network: network["name"].casefold())
