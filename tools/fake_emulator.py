"""
Stand-in for the emulator, for testing the bot without the game.

Start the bot first (python PythonAPI/run_bot.py 1), then:

    python tools/fake_emulator.py --side 1 --frames 3000 --p1 0 --p2 7

It sends synthetic game states over the same socket
protocol and measures the bot's round-trip time. It does NOT simulate the
game's reaction to the buttons - it only proves the bot connects, answers
valid commands every frame, logs, and is fast enough.
"""
import argparse
import json
import random
import socket
import time

BTN = {'Up': False, 'Down': False, 'Left': False, 'Right': False, 'Select': False, 'Start': False,
       'Y': False, 'B': False, 'X': False, 'A': False, 'L': False, 'R': False}


def synthetic(n, c1, c2):
    x1, x2, h1, h2 = 200, 300, 176, 176
    for i in range(n):
        x1 = max(30, min(500, x1 + random.choice([-2, 0, 2])))
        x2 = max(30, min(500, x2 + random.choice([-2, 0, 2])))
        if i > 0 and i % 900 == 0:
            x1, x2 = x2, x1                       # swap sides now and then
        if random.random() < 0.01:
            h2 = max(0, h2 - 8)
        yield {
            'timer': 0x99 - (i // 60) % 99, 'result': 'NOT_OVER', 'round_started': True, 'round_over': False,
            'p1': {'character': c1, 'health': h1, 'x': x1, 'y': 192, 'jumping': False, 'crouching': False,
                   'buttons': BTN, 'in_move': random.random() < 0.2, 'move': random.choice([0, 33686018, 512])},
            'p2': {'character': c2, 'health': h2, 'x': x2, 'y': 192, 'jumping': False, 'crouching': False,
                   'buttons': BTN, 'in_move': random.random() < 0.2, 'move': random.choice([0, 1029, 132101])},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', type=int, default=1)
    ap.add_argument('--frames', type=int, default=3000)
    ap.add_argument('--p1', type=int, default=0)
    ap.add_argument('--p2', type=int, default=7)
    args = ap.parse_args()

    s = socket.create_connection(('127.0.0.1', 9999 if args.side == 1 else 10000))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    dec = json.JSONDecoder()
    rtts, bad = [], 0
    for state in synthetic(args.frames, args.p1, args.p2):
        t0 = time.perf_counter()
        s.sendall(json.dumps(state).encode())
        buf = ''
        while True:
            buf += s.recv(65536).decode()
            try:
                cmd, _ = dec.raw_decode(buf)
                break
            except json.JSONDecodeError:
                continue
        rtts.append((time.perf_counter() - t0) * 1000)
        b = cmd['p1' if args.side == 1 else 'p2']
        if (b['Left'] and b['Right']) or (b['Up'] and b['Down']) or b['Start'] or b['Select']:
            bad += 1
    s.close()
    rtts.sort()
    print(f'{len(rtts)} frames | round-trip median {rtts[len(rtts) // 2]:.2f} ms, '
          f'p99 {rtts[int(len(rtts) * 0.99)]:.2f} ms, worst {rtts[-1]:.2f} ms | invalid commands: {bad}')


if __name__ == '__main__':
    main()
