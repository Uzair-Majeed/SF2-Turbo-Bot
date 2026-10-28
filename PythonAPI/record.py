"""
Record your own gameplay as training data.

    python record.py 1            # you play Player 1 (single-player emulator, vs CPU)
    python record.py 2            # you play Player 2 (two-players emulator)

Same keyboard controls as the kit (arrows, A S D Z X C = Y B X A L R,
Enter = Start, Space = Select). Unlike the kit recorder, EVERY frame of
every round is saved - including frames where you press nothing - so the
model learns spacing, waiting and blocking, and the frame sequence stays
continuous for special-move motions.

Each run writes one file: ../recordings/rec_<date-time>_P<side>.csv
"""
import argparse
import csv
import os
import time
import uuid

from buttons import Buttons
from command import Command
from listen_to_key import get_current_keypress
from net import JsonStream, listen, send_command
import sf2_features as F

REC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recordings'))
FRAME_KEYS = ['timer', 'result', 'round_started', 'round_over',
              'p1_char', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move',
              'p2_char', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move']
BTN_KEYS = ['up', 'down', 'left', 'right', 'y', 'b', 'x', 'a', 'l', 'r']
COLUMNS = ['session', 'frame', 'wall_time', 'side'] + FRAME_KEYS + [f'btn_{k}' for k in BTN_KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('side', type=int, choices=[1, 2], help='which player YOU control')
    args = ap.parse_args()
    side = args.side

    os.makedirs(REC_DIR, exist_ok=True)
    path = os.path.join(REC_DIR, f"rec_{time.strftime('%Y%m%d_%H%M%S')}_P{side}.csv")
    session = uuid.uuid4().hex[:8]
    fh = open(path, 'w', newline='', buffering=1)
    w = csv.writer(fh)
    w.writerow(COLUMNS)

    sock = listen(side)
    stream = JsonStream(sock)
    frame, saved, t_status = 0, 0, time.time()
    try:
        while True:
            msg = stream.latest()
            if msg is None:
                break
            frame += 1
            keys = set(get_current_keypress())

            b = Buttons()
            b.up, b.down, b.left, b.right = 'UP' in keys, 'DOWN' in keys, 'LEFT' in keys, 'RIGHT' in keys
            b.Y, b.B, b.X, b.A, b.L, b.R = ('Y' in keys, 'B' in keys, 'X' in keys,
                                            'A' in keys, 'L' in keys, 'R' in keys)
            b.start, b.select = 'START' in keys, 'SELECT' in keys
            cmd = Command()
            if side == 1:
                cmd.player_buttons = b
            else:
                cmd.player2_buttons = b
            send_command(sock, cmd)

            f = F.frame_from_json(msg)
            if f['round_started']:
                row = [session, frame, f'{time.time():.4f}', side] + [f[k] for k in FRAME_KEYS]
                row += [int(k.upper() in keys) for k in BTN_KEYS]
                w.writerow(row)
                saved += 1

            if time.time() - t_status > 5:
                me = f[f'p{side}_char']
                name = F.CHARACTER_NAMES[me] if 0 <= me < 12 else me
                print(f'[Rec] {saved} frames saved ({saved / 60:.0f} s of fighting) - playing {name}')
                t_status = time.time()
    except (ConnectionResetError, ConnectionAbortedError, KeyboardInterrupt):
        pass
    finally:
        fh.close()
        sock.close()
        print(f'[Rec] saved {saved} frames to {path}')


if __name__ == '__main__':
    main()
