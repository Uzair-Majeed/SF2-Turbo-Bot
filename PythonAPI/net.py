"""
Low-latency socket helpers shared by run_bot.py and record.py.

Speaks exactly the same protocol as the kit's controller.py (the emulator
connects to 127.0.0.1:9999 for P1 / 10000 for P2, sends JSON game states,
and receives JSON commands), but:
  * handles JSON messages split across or packed into one TCP read
  * if several states are waiting, answers only the newest one (no lag build-up)
  * disables Nagle's algorithm so replies are sent immediately
  * prints nothing per frame (console printing is slow on Windows)
"""
import json
import socket

PORTS = {1: 9999, 2: 10000}
_decoder = json.JSONDecoder()


def listen(side):
    port = PORTS[int(side)]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(1)
    print(f'[Net] waiting for the emulator on port {port} - open Gyroscope Bot in the Tool Box')
    client, _ = srv.accept()
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print('Connected to game!')
    return client


class JsonStream:
    def __init__(self, sock):
        self.sock = sock
        self.buf = ''

    def latest(self):
        """Block until at least one full JSON message arrives; return the newest one (None on disconnect)."""
        while True:
            msgs = self._drain()
            if msgs:
                return msgs[-1]
            data = self.sock.recv(65536)
            if not data:
                return None
            self.buf += data.decode('utf-8', errors='replace')

    def _drain(self):
        out = []
        while True:
            s = self.buf.lstrip()
            if not s:
                self.buf = ''
                return out
            try:
                obj, end = _decoder.raw_decode(s)
            except json.JSONDecodeError:
                self.buf = s
                return out
            out.append(obj)
            self.buf = s[end:]


def send_command(sock, cmd):
    sock.sendall(json.dumps(cmd.object_to_dict()).encode())
