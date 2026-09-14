"""Exactly one outstanding request on a framed TCP stream."""
import json
import socket
import time
from pathlib import Path


class BridgeError(RuntimeError):
    pass


class Bridge:
    def __init__(self, port=9999, timeout=120, trace: Path | None = None):
        self.timeout = timeout
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        self.listener.bind(("127.0.0.1", port))
        self.listener.listen(1)
        self.listener.settimeout(timeout)
        self.client = None
        self.buffer = bytearray()
        self.request_id = 0
        self.last_sequence = -1
        self.trace = trace.open("a", encoding="utf-8") if trace else None

    def connect(self):
        if self.client is None:
            self.client, _ = self.listener.accept()
            self.client.settimeout(self.timeout)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hello = self._receive()
            self._validate(hello)
            return hello
        raise BridgeError("Bridge already connected")

    def _receive(self):
        while b"\n" not in self.buffer:
            chunk = self.client.recv(65536)
            if not chunk:
                raise BridgeError("Game disconnected")
            self.buffer.extend(chunk)
            if len(self.buffer) > 4_000_000:
                raise BridgeError("Oversized game message")
        line, _, remaining = self.buffer.partition(b"\n")
        self.buffer = bytearray(remaining)
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise BridgeError("Invalid game JSON") from exc
        if self.trace:
            self.trace.write(json.dumps({"time": time.time(), "direction": "game", "message": value}) + "\n")
            self.trace.flush()
        return value

    def _validate(self, state):
        if state.get("type") != "state" or state.get("protocol") != 1:
            raise BridgeError(f"Unsupported bridge response: {state.get('type')}")
        sequence = state["sequence"]
        if sequence <= self.last_sequence:
            raise BridgeError("Stale or duplicate state sequence")
        self.last_sequence = sequence

    def request(self, op, **fields):
        if self.client is None:
            raise BridgeError("Call connect first")
        self.request_id += 1
        message = dict(op=op, id=self.request_id, **fields)
        self.client.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
        if self.trace:
            self.trace.write(json.dumps({"time": time.time(), "direction": "python", "message": message}) + "\n")
        # A receive timeout is not retried as a new action: that would duplicate it.
        result = self._receive()
        if result.get("id") != self.request_id:
            raise BridgeError("Action/observation request ID mismatch")
        if op != "close":
            self._validate(result)
        return result

    def close(self):
        if self.client:
            try:
                # Shutdown must not inherit the 120-second gameplay timeout.
                # Dead/menu-screen games may never acknowledge another command.
                self.client.settimeout(min(self.timeout,1.0))
                self.request("close")
            except (OSError, BridgeError):
                pass
            self.client.close()
            self.client = None
        self.listener.close()
        if self.trace:
            self.trace.close()
