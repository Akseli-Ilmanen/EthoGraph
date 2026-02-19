"""Phy plugin that listens for commands from EthoGraph via UDP.

Installation:
    1. Copy this file to ~/.phy/plugins/
    2. Add to ~/.phy/phy_config.py:
        c.Plugins.dirs = [r'C:\\Users\\<you>\\.phy\\plugins'] # Windows
        c.TemplateGUI.plugins = ['EthoGraphBridgePlugin']

Protocol:
    EthoGraph sends JSON-encoded UDP packets to localhost:19548.
    Supported commands:
        {"action": "select", "cluster_ids": [42, 17]}
        {"action": "next_spike"}
        {"action": "previous_spike"}
"""

import json
import socket
import threading

from phy import IPlugin, connect
from qtpy.QtCore import QObject, Signal

PHY_BRIDGE_PORT = 19548


class _BridgeSignals(QObject):
    select_signal = Signal(list)
    next_spike_signal = Signal()
    previous_spike_signal = Signal()


class EthoGraphBridgePlugin(IPlugin):

    def attach_to_controller(self, controller):
        self._controller = controller
        self._running = False
        self._thread = None
        self._trace_view = None
        self._signals = _BridgeSignals()
        self._signals.select_signal.connect(self._select_clusters)
        self._signals.next_spike_signal.connect(self._next_spike)
        self._signals.previous_spike_signal.connect(self._previous_spike)

        @connect
        def on_gui_ready(sender, gui):
            self._gui = gui
            self._start_listener()

        @connect
        def on_close(sender):
            self._stop_listener()

        @connect
        def on_view_attached(view, gui):
            if view.__class__.__name__ == 'TraceView':
                self._trace_view = view
                print(f"[EthoGraph bridge] TraceView captured")

    def _start_listener(self):
        if self._running:
            return
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind(("127.0.0.1", PHY_BRIDGE_PORT))
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"[EthoGraph bridge] Listening on UDP port {PHY_BRIDGE_PORT}")

    def _stop_listener(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _listen_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                msg = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            action = msg.get("action")
            if action == "select":
                cluster_ids = msg.get("cluster_ids")
                if cluster_ids:
                    self._signals.select_signal.emit(list(cluster_ids))
            elif action == "next_spike":
                self._signals.next_spike_signal.emit()
            elif action == "previous_spike":
                self._signals.previous_spike_signal.emit()

    def _next_spike(self):
        if self._trace_view is not None:
            self._trace_view.go_to_next_spike()

    def _previous_spike(self):
        if self._trace_view is not None:
            self._trace_view.go_to_previous_spike()

    def _select_clusters(self, cluster_ids):
        self._controller.supervisor.select(cluster_ids)
