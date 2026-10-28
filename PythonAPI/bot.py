"""
Team bot: one TensorFlow model drives every character, from either side.

Launch with run_bot.py (fast path, recommended):
    python run_bot.py 1 --tag stage1      # Player 1
    python run_bot.py 2 --tag stage1      # Player 2

It is also drop-in compatible with the kit's controller.py
(`python controller.py 1` / `python controller.py 2`), which builds
Bot(player_id=...) and calls bot.fight(gs, player_id).

Models: models/model_shared.keras (+ .scaler, .meta.json) is required and
covers all 12 characters. Optional per-character fine-tunes
models/model_<id>.keras (+ .scaler, .meta.json) are used automatically when
that character is being played. All models are loaded and warmed up at
start-up so no frame ever waits on disk or graph compilation.

Every button press comes from the model's output probabilities. The only
post-processing is turning probabilities into buttons and keeping commands
valid under the competition rules (see Decoder below).
"""
import glob
import json
import os
import time
from collections import deque

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import joblib
import numpy as np
import tensorflow as tf

from buttons import Buttons
from command import Command
from device_utils import get_device
from match_logger import MatchLogger
import sf2_features as F

DEBUG = os.environ.get('SF2_DEBUG', '0') == '1'
DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class LoadedModel:
    def __init__(self, path, device):
        self.path = path
        with open(path + '.meta.json') as fh:
            self.meta = json.load(fh)
        if self.meta.get('feature_version') != F.N_FEATURES:
            raise ValueError(f'{path} was trained with a different feature layout - retrain it.')
        scaler = joblib.load(path + '.scaler')
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)
        self.window = int(self.meta['window'])
        self.trained_chars = self.meta.get('trained_chars', {})
        self.trained_opp_chars = self.meta.get('trained_opp_chars', {})
        with tf.device(device):
            self.model = tf.keras.models.load_model(path, compile=False)

        w = self.window
        spec = [tf.TensorSpec([1, w, F.N_FEATURES], tf.float32),
                tf.TensorSpec([1, w, F.N_CAT], tf.int32),
                tf.TensorSpec([1, 2], tf.int32)]

        def make(xla):
            @tf.function(input_signature=spec, jit_compile=xla)
            def predict(x, cat, chars):
                d, a = self.model([x, cat, chars], training=False)
                return d[0], a[0]
            return predict

        dummy = (tf.zeros([1, w, F.N_FEATURES]), tf.zeros([1, w, F.N_CAT], tf.int32),
                 tf.zeros([1, 2], tf.int32))
        with tf.device(device):
            try:
                self.predict = make(True)
                for _ in range(3):
                    self.predict(*dummy)
            except Exception:
                self.predict = make(False)
                for _ in range(3):
                    self.predict(*dummy)


class Decoder:
    """Turns the model's probabilities into one valid button command.

    * direction: most likely of the 9 joystick directions (never opposite pairs)
    * attack: pressed when the model's total attack probability passes a
      threshold; a fresh press samples among attacks in proportion to the
      model's probabilities (move variety), a held press keeps the model's
      top attack, released after MAX_HOLD frames (holding does nothing in SF2)
    * rule 4 (spam): an attack that already has SPAM_LIMIT fresh presses in
      the last 10 s is treated as unavailable and the model's next choice is used
    * rule 5 (idle): after IDLE_LIMIT idle frames the model's most likely
      non-neutral direction is used instead of neutral
    """

    def __init__(self):
        self.atk_threshold = _env_float('SF2_ATK_THRESHOLD', 0.10)
        self.atk_temp = _env_float('SF2_ATK_TEMP', 0.8)
        self.dir_temp = _env_float('SF2_DIR_TEMP', 0.0)
        self.spam_limit = int(_env_float('SF2_SPAM_LIMIT', 6))
        self.idle_limit = int(_env_float('SF2_IDLE_LIMIT', 75))
        self.max_hold = int(_env_float('SF2_MAX_HOLD', 8))
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.presses = {a: deque() for a in range(1, F.N_ATKS)}  # (frame, wall time)
        self.idle = 0
        self.frame = 0
        self.hold = 0

    def _spam_blocked(self, now):
        blocked = set()
        for a, q in self.presses.items():
            while q and (self.frame - q[0][0] > 600 and now - q[0][1] > 10.0):
                q.popleft()
            if len(q) >= self.spam_limit:
                blocked.add(a)
        return blocked

    def decide(self, pdir, patk, prev_atk, active):
        self.frame += 1
        if not active:
            self.idle = 0
            return 0, 0
        now = time.time()

        # ---- direction ----
        if self.dir_temp > 0:
            p = np.power(np.maximum(pdir, 1e-9), 1.0 / self.dir_temp)
            d = int(self.rng.choice(F.N_DIRS, p=p / p.sum()))
        else:
            d = int(np.argmax(pdir))

        # ---- attack ----
        a = 0
        if self.hold >= self.max_hold:
            self.hold = 0                          # release so the next attack is a real new press
        elif 1.0 - float(patk[0]) >= self.atk_threshold:
            probs = np.array(patk[1:], dtype=np.float64)
            if prev_atk and probs.argmax() + 1 == prev_atk:
                a = prev_atk                       # keep holding, not a new press
            else:
                for b in self._spam_blocked(now):
                    probs[b - 1] = 0.0
                if probs.sum() > 0:
                    p = np.power(probs, 1.0 / max(self.atk_temp, 1e-3))
                    a = int(self.rng.choice(F.N_ATKS - 1, p=p / p.sum())) + 1
        if a and a != prev_atk:
            self.presses[a].append((self.frame, now))
        self.hold = self.hold + 1 if (a and a == prev_atk) else (1 if a else 0)

        # ---- idle ----
        if d == 0 and a == 0:
            self.idle += 1
            if self.idle >= self.idle_limit:
                d = int(np.argmax(pdir[1:])) + 1
                self.idle = 0
        else:
            self.idle = 0
        return d, a


class SideState:
    def __init__(self, side, center, window, log_tag):
        self.encoder = F.Encoder(side, center)
        self.buffer = deque(maxlen=window)
        self.decoder = Decoder()
        self.logger = MatchLogger(side, tag=log_tag)


class Bot:
    def __init__(self, player_id=0, model_path=None, model_dir=None, log_tag=None):
        # player_id is accepted for compatibility with the kit's controller.py,
        # which passes Player 1's character id here. The side and characters
        # are read from every frame instead.
        self.device = get_device()
        model_dir = model_dir or (os.path.dirname(model_path) if model_path else DEFAULT_MODEL_DIR)
        self.log_tag = log_tag or os.environ.get('SF2_LOG_TAG', 'match')

        shared = os.path.join(model_dir, 'model_shared.keras')
        if not os.path.exists(shared + '.meta.json'):
            raise FileNotFoundError(f'{shared} (+ .scaler, .meta.json) not found - run train_models/train_shared.py first.')
        self.shared = LoadedModel(shared, self.device)
        self.window = self.shared.window
        self.center = float(self.shared.meta.get('stage_center', F.DEFAULT_STAGE_CENTER))

        self.per_char = {}
        for p in glob.glob(os.path.join(model_dir, 'model_*.keras')):
            name = os.path.basename(p)[len('model_'):-len('.keras')]
            if name.isdigit() and os.path.exists(p + '.meta.json'):
                m = LoadedModel(p, self.device)
                if m.window == self.window:
                    self.per_char[int(name)] = m
        print(f'[Bot] shared model + {len(self.per_char)} character models ready on {self.device} '
              f'(fine-tuned: {sorted(self.per_char) or "none"})')
        self.sides = {}

    # ---- kit controller.py compatibility ----
    def fight(self, gs, player_id):
        t0 = time.perf_counter()
        side = int(player_id)
        buttons = self.act(F.frame_from_gamestate(gs), side, t0)
        return self.make_command(buttons, side)

    @staticmethod
    def make_command(buttons, side):
        b = Buttons()
        b.up, b.down, b.left, b.right = buttons['UP'], buttons['DOWN'], buttons['LEFT'], buttons['RIGHT']
        b.Y, b.B, b.X, b.A, b.L, b.R = (buttons['Y'], buttons['B'], buttons['X'],
                                        buttons['A'], buttons['L'], buttons['R'])
        cmd = Command()
        if side == 1:
            cmd.player_buttons = b
        else:
            cmd.player2_buttons = b
        return cmd

    # ---- main decision ----
    def act(self, f, side, t_recv=None):
        t_recv = t_recv if t_recv is not None else time.perf_counter()
        st = self.sides.get(side)
        if st is None:
            st = self.sides[side] = SideState(side, self.center, self.window, self.log_tag)
        enc = st.encoder

        my_char, opp_char = enc.chars(f)
        m = self.per_char.get(my_char, self.shared)

        x, cat = enc.encode(f)
        if not st.buffer:
            for _ in range(self.window):
                st.buffer.append((x, cat))
        st.buffer.append((x, cat))

        xs = np.stack([b[0] for b in st.buffer])
        xs[:, :F.N_CONT] = (xs[:, :F.N_CONT] - m.mean) / m.scale
        cats = np.stack([b[1] for b in st.buffer])
        chars = np.array([F.char_inputs(my_char, opp_char, m.trained_chars, m.trained_opp_chars)],
                         dtype=np.int32)

        pdir, patk = m.predict(tf.constant(xs[None]), tf.constant(cats[None]), tf.constant(chars))
        pdir, patk = pdir.numpy(), patk.numpy()

        active = bool(f['round_started']) and not bool(f['round_over'])
        d, a = st.decoder.decide(pdir, patk, enc.prev_atk, active)
        enc.commit(d, a)
        buttons = F.decode_action(d, a, enc.facing)

        latency_ms = (time.perf_counter() - t_recv) * 1000.0
        st.logger.log(f, buttons, latency_ms)
        if DEBUG:
            print(f'[Bot] P{side} dir={F.DIRECTIONS[d]} atk={F.ATTACKS[a]} '
                  f'p_atk={1 - patk[0]:.2f} {latency_ms:.1f}ms')
        return buttons

    def close(self):
        for st in self.sides.values():
            st.logger.close()
