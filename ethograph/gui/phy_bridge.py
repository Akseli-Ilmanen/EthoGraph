"""Lightweight UDP client for sending commands to a running Phy instance.

Phy must have the EthoGraphBridgePlugin installed (see extras/phy_ethograph_bridge.py).
All sends are fire-and-forget UDP — if Phy isn't running, packets are silently dropped.
"""

import atexit
import json
import socket

PHY_BRIDGE_PORT = 19548
_PHY_ADDR = ("127.0.0.1", PHY_BRIDGE_PORT)

_sock: socket.socket | None = None


def _get_sock() -> socket.socket:
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        atexit.register(close)
    return _sock


def close() -> None:
    global _sock
    if _sock is not None:
        _sock.close()
        _sock = None


def send_to_phy(command: dict) -> None:
    try:
        payload = json.dumps(command).encode("utf-8")
        _get_sock().sendto(payload, _PHY_ADDR)
    except OSError:
        pass


def phy_next_spike() -> None:
    send_to_phy({"action": "next_spike"})


def phy_previous_spike() -> None:
    send_to_phy({"action": "previous_spike"})


def phy_select_clusters(cluster_ids: list[int]) -> None:
    send_to_phy({"action": "select", "cluster_ids": cluster_ids})
