"""
Automatic per-fight CSV match logs (competition rule 9).

One file per fight, e.g. match_logs/stage1_fight1_P1.csv. A new file is
started when the characters change, or when a new round begins after one
side has already won 2 rounds. Files are never overwritten: the fight number
keeps counting up past existing files. Every row is flushed immediately so a
crash cannot lose the log.
"""
import csv
import os
import re
import time

from sf2_features import GAME_BUTTONS

LOG_COLUMNS = (['frame', 'wall_time', 'side', 'char_id', 'opp_char_id', 'round',
                'round_started', 'round_over', 'result', 'timer',
                'p1_health', 'p2_health', 'self_health', 'opp_health',
                'p1_x', 'p2_x', 'buttons']
               + [f'btn_{b}' for b in GAME_BUTTONS] + ['btn_START', 'btn_SELECT', 'latency_ms'])

DEFAULT_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'match_logs'))


class MatchLogger:
    def __init__(self, side, tag='match', log_dir=DEFAULT_LOG_DIR):
        self.side = int(side)
        self.tag = re.sub(r'[^A-Za-z0-9_-]', '_', tag or 'match')
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._file = None
        self._writer = None
        self._chars = None
        self._wins = {'P1': 0, 'P2': 0}
        self._round = 0
        self._prev_over = False
        self._prev_started = False
        self._frame = 0
        self.path = None

    def _next_path(self):
        pat = re.compile(rf'^{re.escape(self.tag)}_fight(\d+)_P{self.side}\.csv$')
        nums = [int(m.group(1)) for n in os.listdir(self.log_dir) if (m := pat.match(n))]
        n = max(nums, default=0) + 1
        return os.path.join(self.log_dir, f'{self.tag}_fight{n}_P{self.side}.csv')

    def _open(self):
        self.close()
        self.path = self._next_path()
        self._file = open(self.path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(LOG_COLUMNS)
        self._file.flush()
        self._wins = {'P1': 0, 'P2': 0}
        self._round = 0
        print(f'[Log] writing {self.path}')

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def log(self, f, buttons, latency_ms):
        """f: canonical frame dict, buttons: dict of the 10 game buttons sent."""
        me, op = ('p1', 'p2') if self.side == 1 else ('p2', 'p1')
        chars = (f['p1_char'], f['p2_char'])
        started, over = bool(f['round_started']), bool(f['round_over'])

        new_round = started and not over and (self._prev_over or not self._prev_started)
        match_done = max(self._wins.values()) >= 2
        opened = False
        if self._file is None or (chars != self._chars and started) or (new_round and match_done):
            self._open()
            opened = True
        self._chars = chars
        if new_round or (opened and started and not over):
            self._round += 1
        if over and not self._prev_over and f['result'] in self._wins:
            self._wins[f['result']] += 1
        self._prev_over, self._prev_started = over, started

        self._frame += 1
        pressed = [b for b in GAME_BUTTONS if buttons.get(b)]
        row = [self._frame, f'{time.time():.4f}', f'P{self.side}', f[f'{me}_char'], f[f'{op}_char'],
               self._round, int(started), int(over), f['result'], f['timer'],
               f['p1_health'], f['p2_health'], f[f'{me}_health'], f[f'{op}_health'],
               f['p1_x'], f['p2_x'], '+'.join(pressed)]
        row += [int(bool(buttons.get(b))) for b in GAME_BUTTONS]
        row += [0, 0, f'{latency_ms:.3f}']
        self._writer.writerow(row)
