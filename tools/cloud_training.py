"""Notebook helpers. Packaging/audit functions require only the standard library.

The verify command imports TensorFlow only inside the isolated training interpreter.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

NAMES = ['Ryu', 'E. Honda', 'Blanka', 'Guile', 'Ken', 'Chun-Li',
         'Zangief', 'Dhalsim', 'M. Bison', 'Sagat', 'Balrog', 'Vega']
BUTTONS = ['up', 'down', 'left', 'right', 'y', 'b', 'x', 'a', 'l', 'r']


def truth(value):
    value = str(value).strip().lower()
    if value not in {'true', 'false', '0', '1'}:
        raise ValueError(f'Invalid boolean: {value!r}')
    return value in {'true', '1'}


def audit_data(root, smoke=False, include_kit=False):
    """Count eligible raw observations, not generated windows or inferred seconds."""
    root = Path(root)
    if smoke:
        paths = [root / 'normalized_character_datasets/normalized_dataset_0.csv']
    else:
        paths = sorted((root / 'recordings').glob('*.csv'))
        if include_kit:
            paths += sorted((root / 'normalized_character_datasets').glob('*.csv'))
    if not paths:
        raise ValueError('No recordings found. Record locally, rebuild the upload ZIP, then upload it again.')
    counts, files = [0] * 12, []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Rebuild the training ZIP.')
        total, usable = 0, 0
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            legacy = 'player1_id' in columns
            required = {'timer', 'p1_health', 'p1_x', 'p1_y', 'p2_health', 'p2_x', 'p2_y',
                        'p1_jumping', 'p1_crouching', 'p1_in_move',
                        'p2_jumping', 'p2_crouching', 'p2_in_move'}
            required |= ({'fight_result', 'has_round_started', 'is_round_over',
                          'player1_id', 'player2_id', 'p1_move_id', 'p2_move_id'}
                         if legacy else {'session', 'frame', 'side', 'result', 'round_started',
                                         'round_over', 'p1_char', 'p2_char', 'p1_move', 'p2_move'})
            prefix = 'player1_buttons_' if legacy else 'btn_'
            required |= {prefix + b for b in BUTTONS}
            if required - columns:
                raise ValueError(f'{path.name}: missing columns {sorted(required - columns)}')
            for row in reader:
                total += 1
                side = 1 if legacy else int(row['side'])
                if side not in (1, 2):
                    raise ValueError(f'{path.name}, row {total}: side must be 1 or 2')
                cid = int(row['player1_id'] if legacy else row[f'p{side}_char'])
                if cid not in range(12):
                    continue  # menus may have invalid character ids
                started = truth(row['has_round_started' if legacy else 'round_started'])
                over = truth(row['is_round_over' if legacy else 'round_over'])
                hp = int(row[f'p{side}_health'])
                for b in BUTTONS:
                    truth(row[prefix + b])
                if started and not over and 0 < hp <= 200:
                    counts[cid] += 1
                    usable += 1
        files.append({'path': path.relative_to(root).as_posix(), 'rows': total,
                      'usable_observations': usable,
                      'source_type': 'starter_format' if legacy else 'team_recording_format',
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'counts_by_controlled_character': dict(zip(NAMES, counts)), 'files': files,
              'note': 'Counts are raw eligible observations, not independent samples, training windows, or seconds.'}
    for cid, (name, count) in enumerate(zip(NAMES, counts)):
        print(f'{cid:2}  {name:10}  {count:8,} usable observations')
    print(result['note'])
    return result


def require_coverage(audit, smoke=False):
    counts = list(audit['counts_by_controlled_character'].values())
    if smoke:
        if sum(counts) < 1500:
            raise ValueError('Too little sample data for this training check.')
    else:
        missing = [NAMES[c] for c in range(12) if counts[c] < 2000]
        if missing:
            raise ValueError('Record more gameplay while CONTROLLING these characters: '
                             + ', '.join(missing)
                             + '. This notebook requires 2,000 eligible observations each as a basic preflight '
                               'check, not a performance target. Use notebook 01 for a Ryu-only check.')


def run_logged(command, cwd, log_path):
    """Keep training output visible and retain it, including when training fails."""
    with Path(log_path).open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                returncode = proc.wait()
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
    if returncode:
        raise RuntimeError(f'Training exited with code {returncode}. See {log_path}; do not export as successful.')


def snapshot_sources(root, run_dir):
    for relative in ['train_models/train_shared.py', 'PythonAPI/sf2_features.py',
                     'tools/cloud_training.py', 'requirements-training.txt', 'DATA_SOURCES.md']:
        source = Path(root) / relative
        if source.exists():
            target = Path(run_dir) / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def verify_models(model_dir, all_characters=False, main_char=0):
    import joblib
    import numpy as np
    import tensorflow as tf
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'PythonAPI'))
    import sf2_features as F

    model_dir = Path(model_dir)
    if not (model_dir / 'model_shared.keras').exists():
        raise FileNotFoundError('Training did not produce model_shared.keras')
    if all_characters and not (model_dir / f'model_{main_char}.keras').exists():
        raise FileNotFoundError('The main-character fine-tune is missing. Check training output.')
    results = {}
    for path in sorted(model_dir.glob('model_*.keras')):
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
        scaler = joblib.load(str(path) + '.scaler')
        if meta['feature_version'] != F.N_FEATURES:
            raise ValueError('Model feature layout does not match the uploaded source.')
        if scaler.mean_.shape != (F.N_CONT,) or scaler.scale_.shape != (F.N_CONT,):
            raise ValueError('Scaler shape does not match the feature encoder.')
        if not (np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()
                and (scaler.scale_ > 0).all()):
            raise ValueError('Scaler contains invalid values.')
        missing = [cid for cid in range(12) if int(meta['trained_chars'].get(str(cid), 0)) < 1500]
        if all_characters and path.name == 'model_shared.keras' and missing:
            raise ValueError(f'Training-window coverage is insufficient for IDs {missing}; record more.')
        model = tf.keras.models.load_model(path, compile=False)
        window = int(meta['window'])
        for cid in range(12):
            x = np.zeros((1, window, F.N_FEATURES), np.float32)
            x[:, :, :F.N_CONT] = (x[:, :, :F.N_CONT] - scaler.mean_) / scaler.scale_
            cat = np.zeros((1, window, F.N_CAT), np.int32)
            chars = np.array([F.char_inputs(cid, (cid + 1) % 12, meta['trained_chars'],
                                           meta.get('trained_opp_chars', {}))], np.int32)
            outputs = model([x, cat, chars], training=False)
            if len(outputs) != 2:
                raise ValueError('Expected direction and attack output heads.')
            for output, width in zip(outputs, [F.N_DIRS, F.N_ATKS]):
                values = output.numpy()
                if values.shape != (1, width) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid output from {path.name} for character {cid}')
        results[path.name] = {'load_and_prediction': 'passed', 'generic_character_ids': missing}
    results['scope'] = 'Artifact loading and synthetic predictions only; not gameplay, win rate, or rule compliance.'
    (model_dir.parent / 'artifact_checks.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', type=Path, required=True)
    parser.add_argument('--all-characters', action='store_true')
    parser.add_argument('--main-char', type=int, default=0, choices=range(12))
    args = parser.parse_args()
    verify_models(args.verify, args.all_characters, args.main_char)
