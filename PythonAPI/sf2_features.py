"""
Shared feature encoding for the team bot - used by BOTH training and live play.

Everything that turns a raw game frame into model inputs (or a model output
back into buttons) lives here, so training and inference can never drift
apart. If you change anything in this file you must retrain.

Key design choices
------------------
* Self/opponent perspective: features are built from "my" side, not P1/P2.
  The same model therefore plays as Player 1 or Player 2.
* Side (left/right) invariance: LEFT/RIGHT are converted to FORWARD/BACK
  relative to the opponent, and x positions are mirrored. A model trained
  while standing on the left plays identically when standing on the right.
* Character awareness: own and opponent character IDs are model inputs, so
  one shared model covers all 12 characters. Characters with too little
  training data are mapped to an UNKNOWN token (a generic fighter policy).
* Action history: the last actions the bot pressed are inputs, so the model
  can learn multi-frame inputs (special-move motions, combos) from your data.
* Outputs are two softmax heads: a 9-way joystick direction and a 7-way
  attack button. Opposite directions (Left+Right, Up+Down) are impossible by
  construction, and Start/Select are never produced.
"""
import numpy as np

CHARACTER_NAMES = ['Ryu', 'E. Honda', 'Blanka', 'Guile', 'Ken', 'Chun-Li',
                   'Zangief', 'Dhalsim', 'M. Bison', 'Sagat', 'Balrog', 'Vega']
N_CHARS = 12
UNKNOWN_CHAR = 12          # embedding slot for characters without enough data
CHAR_VOCAB = N_CHARS + 1

# ---- joystick directions, relative to the opponent ----
DIRECTIONS = ['NEUTRAL', 'UP', 'DOWN', 'FWD', 'BACK',
              'UP_FWD', 'UP_BACK', 'DOWN_FWD', 'DOWN_BACK']
N_DIRS = len(DIRECTIONS)
_DIR_INDEX = {  # (vertical, horizontal) -> index ; vertical: 1 up, -1 down ; horizontal: 1 fwd, -1 back
    (0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4,
    (1, 1): 5, (1, -1): 6, (-1, 1): 7, (-1, -1): 8,
}
_DIR_VECTOR = {v: k for k, v in _DIR_INDEX.items()}

# ---- attack buttons (SNES SF2 Turbo layout) ----
ATTACKS = ['NONE', 'LP', 'MP', 'HP', 'LK', 'MK', 'HK']
N_ATKS = len(ATTACKS)
ATTACK_BUTTON = {1: 'Y', 2: 'X', 3: 'L', 4: 'B', 5: 'A', 6: 'R'}
# when a human pressed several attack buttons at once, keep the strongest
_ATTACK_PRIORITY = [(3, 'l'), (6, 'r'), (2, 'x'), (5, 'a'), (1, 'y'), (4, 'b')]

GAME_BUTTONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'Y', 'B', 'X', 'A', 'L', 'R']

# ---- per-frame feature layout ----
CONT_FEATURES = ['timer', 'self_hp', 'opp_hp', 'hp_diff', 'dist', 'dy',
                 'self_xr', 'opp_xr', 'self_y', 'opp_y', 'since_atk', 'opp_since_move']
BIN_FEATURES = (['round_started', 'round_over',
                 'self_jump', 'self_crouch', 'self_in_move',
                 'opp_jump', 'opp_crouch', 'opp_in_move']
                + [f'prev_dir_{d}' for d in DIRECTIONS]
                + [f'prev_atk_{a}' for a in ATTACKS])
N_CONT = len(CONT_FEATURES)          # these columns go through the StandardScaler
N_FEATURES = N_CONT + len(BIN_FEATURES)
PREV_ACTION_START = N_CONT + 8       # first prev_dir/prev_atk column
N_CAT = 8                            # 4 move-id bytes for self + 4 for opponent
CAT_VOCAB = N_CAT * 256              # byte value offset by its position

SINCE_CAP = 120
DEFAULT_STAGE_CENTER = 256.0
X_MIN, X_MAX = 0, 1024


# ------------------------------------------------------------------
# raw frame normalisation
# ------------------------------------------------------------------
def _b(v):
    if isinstance(v, str):
        return 1 if v.strip().lower() in ('true', '1') else 0
    return 1 if v else 0


def frame_from_json(d):
    """Flatten the emulator's JSON game state into the canonical frame dict."""
    p1, p2 = d['p1'], d['p2']
    return {
        'timer': d['timer'], 'result': d['result'],
        'round_started': _b(d['round_started']), 'round_over': _b(d['round_over']),
        'p1_char': p1['character'], 'p1_health': p1['health'], 'p1_x': p1['x'], 'p1_y': p1['y'],
        'p1_jumping': _b(p1['jumping']), 'p1_crouching': _b(p1['crouching']),
        'p1_in_move': _b(p1['in_move']), 'p1_move': p1['move'],
        'p2_char': p2['character'], 'p2_health': p2['health'], 'p2_x': p2['x'], 'p2_y': p2['y'],
        'p2_jumping': _b(p2['jumping']), 'p2_crouching': _b(p2['crouching']),
        'p2_in_move': _b(p2['in_move']), 'p2_move': p2['move'],
    }


def frame_from_gamestate(gs):
    """Same as frame_from_json but from the kit's GameState object."""
    out = {'timer': gs.timer, 'result': gs.fight_result,
           'round_started': _b(gs.has_round_started), 'round_over': _b(gs.is_round_over)}
    for tag, p in (('p1', gs.player1), ('p2', gs.player2)):
        out.update({
            f'{tag}_char': p.player_id, f'{tag}_health': p.health,
            f'{tag}_x': p.x_coord, f'{tag}_y': p.y_coord,
            f'{tag}_jumping': _b(p.is_jumping), f'{tag}_crouching': _b(p.is_crouching),
            f'{tag}_in_move': _b(p.is_player_in_move), f'{tag}_move': p.move_id,
        })
    return out


def decode_timer(t):
    """The game stores the round clock as BCD (0x99 = 153 -> 99 seconds)."""
    t = int(t)
    hi, lo = (t >> 4) & 0xF, t & 0xF
    if 0 <= t <= 0x99 and hi <= 9 and lo <= 9:
        return hi * 10 + lo
    return min(max(t, 0), 99)


def clean_health(h):
    h = int(h)
    return 0 if (h < 0 or h > 200) else h   # 255 (= -1) appears on KO


def clean_x(x):
    return float(min(max(int(x), X_MIN), X_MAX))


def move_bytes(m):
    m = int(m) & 0xFFFFFFFF
    return [(m >> 24) & 255, (m >> 16) & 255, (m >> 8) & 255, m & 255]


# ------------------------------------------------------------------
# actions
# ------------------------------------------------------------------
def encode_action(btn, facing):
    """btn: dict with lowercase keys up/down/left/right/y/b/x/a/l/r -> (dir_idx, atk_idx)."""
    v = (1 if btn.get('up') else 0) - (1 if btn.get('down') else 0)
    h_abs = (1 if btn.get('right') else 0) - (1 if btn.get('left') else 0)
    h = h_abs * (1 if facing >= 0 else -1)
    atk = 0
    for idx, key in _ATTACK_PRIORITY:
        if btn.get(key):
            atk = idx
            break
    return _DIR_INDEX[(v, h)], atk


def decode_action(dir_idx, atk_idx, facing):
    """(dir_idx, atk_idx, facing) -> dict of the 10 game buttons (UPPERCASE keys)."""
    v, h = _DIR_VECTOR[int(dir_idx)]
    h_abs = h * (1 if facing >= 0 else -1)
    out = {b: False for b in GAME_BUTTONS}
    out['UP'] = v == 1
    out['DOWN'] = v == -1
    out['RIGHT'] = h_abs == 1
    out['LEFT'] = h_abs == -1
    if atk_idx:
        out[ATTACK_BUTTON[int(atk_idx)]] = True
    return out


def is_movement(dir_idx):
    return int(dir_idx) != 0


# ------------------------------------------------------------------
# stateful per-side encoder
# ------------------------------------------------------------------
class Encoder:
    """Turns successive frames into feature vectors from one side's point of view.

    Call encode(frame) for each frame, then commit(dir, atk) with the action
    actually taken on that frame (the human's in training, the bot's live).
    """

    def __init__(self, side, stage_center=DEFAULT_STAGE_CENTER):
        self.side = int(side)
        self.me = 'p1' if self.side == 1 else 'p2'
        self.op = 'p2' if self.side == 1 else 'p1'
        self.center = float(stage_center)
        self.reset()

    def reset(self):
        self.facing = 1 if self.side == 1 else -1
        self.prev_dir = 0
        self.prev_atk = 0
        self.since_atk = SINCE_CAP
        self.opp_since_move = SINCE_CAP
        self._opp_prev_in_move = 0

    def encode(self, f):
        me, op = self.me, self.op
        sx, ox = clean_x(f[f'{me}_x']), clean_x(f[f'{op}_x'])
        dx = ox - sx
        if dx > 0:
            self.facing = 1
        elif dx < 0:
            self.facing = -1
        fc = self.facing
        sy, oy = float(f[f'{me}_y']), float(f[f'{op}_y'])
        shp, ohp = clean_health(f[f'{me}_health']), clean_health(f[f'{op}_health'])

        opp_in_move = _b(f[f'{op}_in_move'])
        if opp_in_move and not self._opp_prev_in_move:
            self.opp_since_move = 0
        else:
            self.opp_since_move = min(self.opp_since_move + 1, SINCE_CAP)
        self._opp_prev_in_move = opp_in_move

        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = decode_timer(f['timer'])
        x[1] = shp
        x[2] = ohp
        x[3] = shp - ohp
        x[4] = abs(dx)
        x[5] = oy - sy
        x[6] = (sx - self.center) * fc
        x[7] = (ox - self.center) * fc
        x[8] = sy
        x[9] = oy
        x[10] = self.since_atk
        x[11] = self.opp_since_move
        j = N_CONT
        x[j + 0] = _b(f['round_started'])
        x[j + 1] = _b(f['round_over'])
        x[j + 2] = _b(f[f'{me}_jumping'])
        x[j + 3] = _b(f[f'{me}_crouching'])
        x[j + 4] = _b(f[f'{me}_in_move'])
        x[j + 5] = _b(f[f'{op}_jumping'])
        x[j + 6] = _b(f[f'{op}_crouching'])
        x[j + 7] = opp_in_move
        x[PREV_ACTION_START + self.prev_dir] = 1.0
        x[PREV_ACTION_START + N_DIRS + self.prev_atk] = 1.0

        cat = np.empty(N_CAT, dtype=np.int32)
        cat[:4] = move_bytes(f[f'{me}_move'])
        cat[4:] = move_bytes(f[f'{op}_move'])
        cat += np.arange(N_CAT, dtype=np.int32) * 256
        return x, cat

    def commit(self, dir_idx, atk_idx):
        atk_idx = int(atk_idx)
        if atk_idx and atk_idx != self.prev_atk:
            self.since_atk = 0
        else:
            self.since_atk = min(self.since_atk + 1, SINCE_CAP)
        self.prev_dir = int(dir_idx)
        self.prev_atk = atk_idx

    def chars(self, f):
        return int(f[f'{self.me}_char']), int(f[f'{self.op}_char'])


def char_token(cid, trained_chars):
    """Map a character id to its embedding slot (UNKNOWN if the model never really saw it)."""
    cid = int(cid)
    if 0 <= cid < N_CHARS and str(cid) in trained_chars:
        return cid
    return UNKNOWN_CHAR


CHAR_INPUT_VOCAB = 2 * CHAR_VOCAB   # own tokens [0, 13), opponent tokens [13, 26)


def char_inputs(my_char, opp_char, trained_self, trained_opp):
    """The model's 2-int character input: [own token, opponent token + CHAR_VOCAB]."""
    return [char_token(my_char, trained_self), char_token(opp_char, trained_opp) + CHAR_VOCAB]
