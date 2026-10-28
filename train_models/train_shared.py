"""
Train the team's shared model (all characters, both sides) and optionally
fine-tune extra per-character models.

Run from gamebot-competition-master:

    python train_models/train_shared.py --main-char 0
    python train_models/train_shared.py --main-char 0 --finetune 0

Inputs : recordings/*.csv (from PythonAPI/record.py)
         + normalized_character_datasets/*.csv with --include-kit-data
Outputs: models/model_shared.keras  (+ .scaler, .meta.json)
         models/model_<id>.keras    (+ .scaler, .meta.json) for each --finetune id
         models/training_report.json

Copy the whole models/ folder back to the match computer afterwards.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'PythonAPI'))
import sf2_features as F  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, callbacks  # noqa: E402

FRAME_KEYS = ['timer', 'result', 'round_started', 'round_over',
              'p1_char', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move',
              'p2_char', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move']
BTN_KEYS = ['up', 'down', 'left', 'right', 'y', 'b', 'x', 'a', 'l', 'r']
BOOL_KEYS = {'round_started', 'round_over', 'p1_jumping', 'p1_crouching', 'p1_in_move',
             'p2_jumping', 'p2_crouching', 'p2_in_move'} | {f'btn_{k}' for k in BTN_KEYS}
LEGACY_RENAME = {
    'fight_result': 'result', 'has_round_started': 'round_started', 'is_round_over': 'round_over',
    'player1_id': 'p1_char', 'p1_move_id': 'p1_move', 'player2_id': 'p2_char', 'p2_move_id': 'p2_move',
    **{f'player1_buttons_{k}': f'btn_{k}' for k in BTN_KEYS},
}


# ------------------------------------------------------------------
# loading
# ------------------------------------------------------------------
def _to_int(s):
    if s.dtype == bool:
        return s.astype(np.int64)
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(np.int64)
    return s.fillna(0).astype(np.int64)


def load_file(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'player1_id' in df.columns:                        # kit (legacy) format
        df = df.rename(columns=LEGACY_RENAME)
        df['session'] = os.path.basename(path)
        df['frame'] = np.arange(len(df))
        df['side'] = 1
        df = df[df['round_started'].astype(str).str.lower().isin(['true', '1'])]
    missing = [c for c in FRAME_KEYS + [f'btn_{k}' for k in BTN_KEYS] if c not in df.columns]
    if missing:
        print(f'  ! skipping {os.path.basename(path)}: missing columns {missing}')
        return None
    for c in BOOL_KEYS:
        df[c] = _to_int(df[c])
    df['result'] = df['result'].astype(str)
    df['session'] = df['session'].astype(str) + '|' + os.path.basename(path)
    df['source'] = os.path.basename(path)
    return df.reset_index(drop=True)


def split_segments(df):
    """Continuous runs of frames (same session, consecutive frame numbers)."""
    brk = (df['session'] != df['session'].shift()) | (df['frame'].diff() != 1)
    seg = brk.cumsum()
    return [g for _, g in df.groupby(seg, sort=False)]


def estimate_center(dfs):
    xs = []
    for df in dfs:
        live = df[(df['round_started'] == 1) & (df['round_over'] == 0)]
        xs += [live['p1_x'].values, live['p2_x'].values]
    xs = np.concatenate(xs) if xs else np.array([])
    xs = xs[(xs > F.X_MIN) & (xs < F.X_MAX)]
    if len(xs) < 1000:
        return F.DEFAULT_STAGE_CENTER
    lo, hi = np.percentile(xs, [1, 99])
    return float((lo + hi) / 2)


# ------------------------------------------------------------------
# encoding
# ------------------------------------------------------------------
def encode(dfs, center, window, val_frac):
    X, C, MY, OP, YD, YA, OK = [], [], [], [], [], [], []
    train_idx, val_idx = [], []
    n = 0
    for df in dfs:
        for seg in split_segments(df):
            side = int(seg['side'].iloc[0])
            enc = F.Encoder(side, center)
            me = 'p1' if side == 1 else 'p2'
            start = n
            for r in seg.to_dict('records'):
                x, cat = enc.encode(r)
                d, a = F.encode_action({k: r[f'btn_{k}'] for k in BTN_KEYS}, enc.facing)
                enc.commit(d, a)
                my, op = enc.chars(r)
                ok = (r['round_started'] == 1 and r['round_over'] == 0
                      and F.clean_health(r[f'{me}_health']) > 0 and 0 <= my < F.N_CHARS)
                X.append(x); C.append(cat); MY.append(my); OP.append(op)
                YD.append(d); YA.append(a); OK.append(ok)
                n += 1
            ends = [i for i in range(start + window - 1, n) if OK[i]]
            cut = int(len(ends) * (1 - val_frac)) if len(ends) >= 50 else len(ends)
            train_idx += ends[:cut]
            val_idx += ends[cut:]
    return (np.asarray(X, np.float32), np.asarray(C, np.int32), np.asarray(MY, np.int32),
            np.asarray(OP, np.int32), np.asarray(YD, np.int32), np.asarray(YA, np.int32),
            np.asarray(train_idx, np.int64), np.asarray(val_idx, np.int64))


# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------
def build_model(window, hidden=128):
    x_in = layers.Input((window, F.N_FEATURES), name='state')
    c_in = layers.Input((window, F.N_CAT), dtype='int32', name='moves')
    ch_in = layers.Input((2,), dtype='int32', name='chars')

    mv = layers.Embedding(F.CAT_VOCAB, 4, name='move_byte_emb')(c_in)
    mv = layers.Reshape((window, F.N_CAT * 4))(mv)
    ch = layers.Flatten()(layers.Embedding(F.CHAR_INPUT_VOCAB, 8, name='char_emb')(ch_in))
    ch_seq = layers.RepeatVector(window)(ch)

    h = layers.Concatenate()([x_in, mv, ch_seq])
    h = layers.Dense(hidden, activation='relu')(h)
    h = layers.Dense(hidden, activation='relu')(h)
    h = layers.GRU(hidden)(h)
    h = layers.Concatenate()([h, ch])
    h = layers.Dense(hidden, activation='relu')(h)
    h = layers.Dropout(0.2)(h)
    out_dir = layers.Dense(F.N_DIRS, activation='softmax', name='dir')(h)
    out_atk = layers.Dense(F.N_ATKS, activation='softmax', name='atk')(h)
    return tf.keras.Model([x_in, c_in, ch_in], [out_dir, out_atk])


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy(),
                        tf.keras.losses.SparseCategoricalCrossentropy()],
                  metrics=[['accuracy'], ['accuracy']])


def class_weights(y, n, power, cap):
    cnt = np.bincount(y, minlength=n).astype(np.float64) + 1
    w = (cnt / cnt.max()) ** (-power)
    return np.minimum(w, cap)


def make_dataset(tensors, idx, window, batch, train, char_drop, prev_drop, noise):
    X, C, CH, YD, YA, WD, WA = tensors
    offs = tf.range(-window + 1, 1, dtype=tf.int64)
    prev_mask = np.zeros(F.N_FEATURES, np.float32)
    prev_mask[F.PREV_ACTION_START:] = 1.0
    prev_mask = tf.constant(prev_mask)
    cont_mask = np.zeros(F.N_FEATURES, np.float32)
    cont_mask[:F.N_CONT] = 1.0
    cont_mask = tf.constant(cont_mask)

    def gather(i):
        w = i[:, None] + offs[None, :]
        x = tf.gather(X, w)
        c = tf.gather(C, w)
        ch = tf.gather(CH, i)
        if train:
            b = tf.shape(i)[0]
            drop = tf.cast(tf.random.uniform([b, 1, 1]) < prev_drop, tf.float32)
            x = x * (1.0 - drop * prev_mask)
            x = x + tf.random.normal(tf.shape(x), stddev=noise) * cont_mask
            unk = tf.constant([F.UNKNOWN_CHAR, F.UNKNOWN_CHAR + F.CHAR_VOCAB], tf.int32)
            dmask = tf.random.uniform([b, 2]) < char_drop
            ch = tf.where(dmask, tf.broadcast_to(unk, [b, 2]), ch)
        return ((x, c, ch), (tf.gather(YD, i), tf.gather(YA, i)), (tf.gather(WD, i), tf.gather(WA, i)))

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if train:
        ds = ds.shuffle(len(idx), reshuffle_each_iteration=True)
    return ds.batch(batch).map(gather, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def attack_report(model, ds, threshold=0.4):
    tp = fp = fn = 0
    for (x, c, ch), (yd, ya), _ in ds:
        _, pa = model((x, c, ch), training=False)
        pred = (1.0 - pa[:, 0].numpy()) >= threshold
        true = ya.numpy() > 0
        tp += int(np.sum(pred & true)); fp += int(np.sum(pred & ~true)); fn += int(np.sum(~pred & true))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {'attack_precision': round(prec, 3), 'attack_recall': round(rec, 3)}


def save_bundle(model, path, scaler, meta):
    model.save(path)
    joblib.dump(scaler, path + '.scaler')
    with open(path + '.meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'  saved {path} (+ .scaler, .meta.json)')


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--main-char', type=int, default=0, help='your Stage 1 character id (gets extra weight)')
    ap.add_argument('--main-weight', type=float, default=2.0)
    ap.add_argument('--include-kit-data', action='store_true', help='also use normalized_character_datasets/*.csv')
    ap.add_argument('--data', nargs='*', default=None, help='explicit CSV files/globs (default: recordings/*.csv)')
    ap.add_argument('--window', type=int, default=12, help='frames of history the model sees')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--min-char-frames', type=int, default=1500,
                    help='characters with fewer frames use the generic UNKNOWN embedding')
    ap.add_argument('--char-dropout', type=float, default=0.15)
    ap.add_argument('--prev-dropout', type=float, default=0.3)
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--finetune', type=int, nargs='*', default=[], help='character ids to fine-tune separately')
    ap.add_argument('--finetune-epochs', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- data ----
    patterns = args.data if args.data else [os.path.join(ROOT, 'recordings', '*.csv')]
    if args.include_kit_data:
        patterns.append(os.path.join(ROOT, 'normalized_character_datasets', '*.csv'))
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not files:
        sys.exit('No training data found. Record some with: python PythonAPI/record.py 1')
    print(f'Loading {len(files)} file(s)...')
    dfs = [d for d in (load_file(p) for p in files) if d is not None and len(d)]
    center = estimate_center(dfs)
    print(f'Stage center estimate: {center:.1f}')

    t0 = time.time()
    X, C, MY, OP, YD, YA, tr, va = encode(dfs, center, args.window, args.val_frac)
    print(f'Encoded {len(X)} frames in {time.time() - t0:.0f}s -> {len(tr)} train / {len(va)} val samples')
    if len(tr) < 1000:
        sys.exit('Too little data to train (need at least ~1000 usable frames).')

    # ---- per-character coverage ----
    my_counts = np.bincount(MY[tr], minlength=F.N_CHARS)
    op_counts = np.bincount(np.clip(OP[tr], 0, F.N_CHARS - 1), minlength=F.N_CHARS)
    trained = {str(c): int(my_counts[c]) for c in range(F.N_CHARS) if my_counts[c] >= args.min_char_frames}
    trained_opp = {str(c): int(op_counts[c]) for c in range(F.N_CHARS) if op_counts[c] >= args.min_char_frames}
    print('\nFrames per character you played (train):')
    for c in range(F.N_CHARS):
        tag = 'own embedding' if str(c) in trained else 'generic (UNKNOWN) embedding'
        print(f'  {c:2d} {F.CHARACTER_NAMES[c]:9s} {my_counts[c]:8d}  -> {tag}')
    print(f'Attack frames: {np.mean(YA[tr] > 0):.1%}   neutral-direction frames: {np.mean(YD[tr] == 0):.1%}')

    # ---- scaling ----
    scaler = StandardScaler()
    scaler.fit(X[tr][:, :F.N_CONT])
    Xs = X.copy()
    Xs[:, :F.N_CONT] = (X[:, :F.N_CONT] - scaler.mean_) / scaler.scale_
    CH = np.array([F.char_inputs(m, o, trained, trained_opp) for m, o in zip(MY, OP)], np.int32)

    # ---- weights ----
    wd_c = class_weights(YD[tr], F.N_DIRS, 0.3, 5.0)
    wa_c = class_weights(YA[tr], F.N_ATKS, 0.5, 10.0)
    base = np.where(MY == args.main_char, args.main_weight, 1.0)
    WD = (base * wd_c[YD]).astype(np.float32)
    WA = (base * wa_c[YA]).astype(np.float32)
    WD /= WD[tr].mean(); WA /= WA[tr].mean()

    tensors = tuple(tf.constant(a) for a in (Xs, C, CH, YD, YA, WD, WA))
    mk = dict(window=args.window, batch=args.batch, char_drop=args.char_dropout,
              prev_drop=args.prev_dropout, noise=args.noise)
    ds_tr = make_dataset(tensors, tr, train=True, **mk)
    ds_va = make_dataset(tensors, va, train=False, **mk) if len(va) else None

    # ---- shared model ----
    model = build_model(args.window, args.hidden)
    compile_model(model, args.lr)
    model.summary()
    shared_path = os.path.join(args.out, 'model_shared.keras')
    tmp_best = os.path.join(args.out, '_best_tmp.keras')
    mon = 'val_loss' if ds_va is not None else 'loss'
    cbs = [callbacks.ModelCheckpoint(tmp_best, monitor=mon, save_best_only=True),
           callbacks.EarlyStopping(monitor=mon, patience=6, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor=mon, factor=0.5, patience=3, min_lr=1e-5)]
    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=cbs, verbose=2)
    if os.path.exists(tmp_best):
        os.remove(tmp_best)

    meta = {
        'feature_version': F.N_FEATURES, 'window': args.window, 'stage_center': center,
        'trained_chars': trained, 'trained_opp_chars': trained_opp, 'main_char': args.main_char,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'tensorflow': tf.__version__,
        'data_files': [os.path.basename(f) for f in files], 'train_samples': int(len(tr)),
        'finetuned_for': None,
    }
    save_bundle(model, shared_path, scaler, meta)
    report = {'shared': attack_report(model, ds_va) if ds_va is not None else {}}
    print('Validation (shared):', report['shared'])

    # ---- optional per-character fine-tunes ----
    rng = np.random.default_rng(args.seed)
    for cid in args.finetune:
        own_tr = tr[MY[tr] == cid]
        own_va = va[MY[va] == cid]
        if len(own_tr) < 1000:
            print(f'\nSkipping fine-tune for {cid} ({F.CHARACTER_NAMES[cid]}): only {len(own_tr)} frames')
            continue
        others = tr[MY[tr] != cid]
        mix = rng.choice(others, size=min(len(others), len(own_tr) // 3), replace=False) if len(others) else others
        ft_tr = np.concatenate([own_tr, mix])
        print(f'\n=== Fine-tuning character {cid} ({F.CHARACTER_NAMES[cid]}) on {len(ft_tr)} samples ===')
        ft = tf.keras.models.load_model(shared_path, compile=False)
        compile_model(ft, args.lr * 0.2)
        ds_ft = make_dataset(tensors, ft_tr, train=True, **mk)
        ds_fv = make_dataset(tensors, own_va, train=False, **mk) if len(own_va) else None
        m2 = 'val_loss' if ds_fv is not None else 'loss'
        ft.fit(ds_ft, validation_data=ds_fv, epochs=args.finetune_epochs, verbose=2,
               callbacks=[callbacks.EarlyStopping(monitor=m2, patience=4, restore_best_weights=True)])
        p = os.path.join(args.out, f'model_{cid}.keras')
        save_bundle(ft, p, scaler, {**meta, 'finetuned_for': cid})
        if ds_fv is not None:
            report[str(cid)] = attack_report(ft, ds_fv)
            print(f'Validation ({cid}):', report[str(cid)])

    with open(os.path.join(args.out, 'training_report.json'), 'w') as fh:
        json.dump({'meta': meta, 'validation': report}, fh, indent=2)
    print('\nDone. Copy the models/ folder to the match computer.')


if __name__ == '__main__':
    main()
