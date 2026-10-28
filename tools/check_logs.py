"""
Check match logs against the competition rules before the judges do.

    python tools/check_logs.py                     # all files in match_logs/
    python tools/check_logs.py match_logs/stage1_fight1_P1.csv

Per round it reports:
  spam    - most fresh presses of one attack button within any 10 s (limit: < 8)
  idle    - longest stretch with no button held at all (limit: < 3 s)
  invalid - frames with Left+Right, Up+Down, Start or Select
  latency - median / worst response time, and answers slower than 500 ms
Time is measured in game frames (60 per second) and wall-clock; the worse
of the two is reported.
"""
import csv
import glob
import os
import sys

ATTACKS = ['Y', 'B', 'X', 'A', 'L', 'R']
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def check(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f'{os.path.basename(path)}: empty')
        return False
    rounds = {}
    for r in rows:
        if r['round_started'] == '1' and r['round_over'] == '0':
            rounds.setdefault(r['round'], []).append(r)

    lat = sorted(float(r['latency_ms']) for r in rows)
    over = sum(v > 500 for v in lat)
    print(f"\n{os.path.basename(path)}  ({len(rows)} frames, chars {rows[-1]['char_id']} vs {rows[-1]['opp_char_id']})")
    print(f'  latency: median {lat[len(lat) // 2]:.1f} ms, worst {lat[-1]:.1f} ms, >500 ms: {over}')
    ok = over == 0
    for rid, rr in rounds.items():
        presses = {a: [] for a in ATTACKS}
        prev = {a: False for a in ATTACKS}
        idle_f = idle_w = max_idle = 0.0
        idle_start_w = None
        invalid = 0
        for i, r in enumerate(rr):
            t = float(r['wall_time'])
            for a in ATTACKS:
                on = r[f'btn_{a}'] == '1'
                if on and not prev[a]:
                    presses[a].append((i, t))
                prev[a] = on
            if (r['btn_LEFT'] == '1' and r['btn_RIGHT'] == '1') or (r['btn_UP'] == '1' and r['btn_DOWN'] == '1') \
                    or r['btn_START'] == '1' or r['btn_SELECT'] == '1':
                invalid += 1
            if r['buttons']:
                idle_f, idle_start_w = 0, None
            else:
                idle_f += 1
                idle_start_w = idle_start_w if idle_start_w is not None else t
                idle_w = t - idle_start_w
                max_idle = max(max_idle, idle_f / 60.0, idle_w)
        spam = 0
        spam_btn = '-'
        for a, ps in presses.items():
            for j in range(len(ps)):
                k = j
                while k < len(ps) and ((ps[k][0] - ps[j][0]) <= 600 or (ps[k][1] - ps[j][1]) <= 10.0):
                    k += 1
                if k - j > spam:
                    spam, spam_btn = k - j, a
        n_press = sum(len(p) for p in presses.values())
        flags = []
        if spam >= 8:
            flags.append('SPAM')
        if max_idle >= 3.0:
            flags.append('IDLE')
        if invalid:
            flags.append('INVALID')
        ok &= not flags
        mix = ', '.join(f'{a}:{len(p)}' for a, p in presses.items() if p)
        print(f'  round {rid}: {len(rr)} frames | spam max {spam} ({spam_btn}) | idle max {max_idle:.1f}s | '
              f'invalid {invalid} | presses {n_press} [{mix}] {"<-- " + " ".join(flags) if flags else "OK"}')
    return ok


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'match_logs', '*.csv')))
    if not paths:
        sys.exit('No logs found in match_logs/')
    results = [check(p) for p in paths]
    print(f'\n{sum(results)}/{len(results)} files clean')


if __name__ == '__main__':
    main()
