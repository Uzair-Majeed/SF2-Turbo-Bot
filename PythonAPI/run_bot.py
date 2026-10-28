"""
Launch the team bot.

    python run_bot.py 1 --tag stage1     # as Player 1 (port 9999)
    python run_bot.py 2 --tag stage1     # as Player 2 (port 10000)

Models are loaded and warmed up BEFORE the emulator connects, so the first
frame is answered as fast as every other. Match logs go to ../match_logs/.
"""
import argparse
import sys
import time

from bot import Bot
from net import JsonStream, listen, send_command
import sf2_features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('side', type=int, choices=[1, 2], help='1 = Player 1, 2 = Player 2')
    ap.add_argument('--tag', default='match', help='log file prefix, e.g. stage1 -> stage1_fight1_P1.csv')
    ap.add_argument('--model-dir', default=None)
    args = ap.parse_args()

    bot = Bot(model_dir=args.model_dir, log_tag=args.tag)
    sock = listen(args.side)
    stream = JsonStream(sock)
    frames, slow, worst = 0, 0, 0.0
    t_report = time.time()
    try:
        while True:
            msg = stream.latest()
            if msg is None:
                print('[Bot] emulator disconnected')
                break
            t0 = time.perf_counter()
            buttons = bot.act(F.frame_from_json(msg), args.side, t0)
            send_command(sock, Bot.make_command(buttons, args.side))
            ms = (time.perf_counter() - t0) * 1000
            frames += 1
            worst = max(worst, ms)
            slow += ms > 100
            if time.time() - t_report > 30:
                print(f'[Bot] {frames} frames, worst response {worst:.1f} ms, >100ms: {slow}')
                t_report, worst = time.time(), 0.0
    except (ConnectionResetError, ConnectionAbortedError):
        print('[Bot] connection closed by emulator')
    finally:
        bot.close()
        sock.close()


if __name__ == '__main__':
    sys.exit(main())
