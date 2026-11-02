# ===================== Microwave Filter Coupling Matrix Design (PYTHONISTA iPad) =====================

import sys
import traceback
import clipboard
import io
import os
import random
import gc
from types import SimpleNamespace
from collections import deque


def _copy_error(exctype, value, tb):
    err = "".join(traceback.format_exception(exctype, value, tb))
    print(err)
    try:
        clipboard.set(err)
    except Exception:
        pass


sys.excepthook = _copy_error

import ui
import dialogs
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# ===================== CONSTANTS =====================

MAX_UNDO = 8
COND_WARN = 1e10
COND_FAIL = 1e14
MIN_QU, MAX_QU = 50.0, 1e8
MIN_FBW, MAX_FBW = 1e-5, 0.8
DEFAULT_NUM_POINTS = 401
SENS_PERTURB = 0.05          # ±5 % for sensitivity

PLOT_MODES = {
    'loss': dict(name='Insertion Loss', ylabel='Insertion Loss (dB)', color_sim='#00ff88',
                 color_meas='orange', unit='dB', key='s21_db'),
    'phase': dict(name='Relative Phase', ylabel='Relative Phase (deg)', color_sim='#ffaa00',
                  color_meas='magenta', unit='°', key='phase_rel'),
    'gd': dict(name='Group Delay', ylabel='Group Delay (ns)', color_sim='#00ffff',
               color_meas='pink', unit='ns', key='gd_ns'),
}

# (dialog title, state attribute, scale, display format, optional/blank allowed)
LIMIT_FIELDS = [
    ('Start Frequency (MHz)', 'start_freq', 1e6, '{:.0f}', False),
    ('Stop Frequency (MHz)', 'stop_freq', 1e6, '{:.0f}', False),
    ('Return Loss y min (dB)', 'rl_ymin', 1, None, False),
    ('Return Loss y max (dB)', 'rl_ymax', 1, None, False),
    ('Through Loss y min (dB)', 'il_ymin', 1, None, False),
    ('Through Loss y max (dB)', 'il_ymax', 1, None, False),
    ('Phase y min (deg)', 'phase_ymin', 1, None, False),
    ('Phase y max (deg)', 'phase_ymax', 1, None, False),
    ('Group Delay y min (ns)', 'gd_ymin', 1, None, True),
    ('Group Delay y max (ns)', 'gd_ymax', 1, None, True),
]

TZ_TITLE = 'Transmission zeros (MHz, comma sep)'

HELP_TEXT = """COUPLING MATRIX TUNING TOOL – HELP

GETTING STARTED
1. Design → set n, f0, BW, Qu, ripple (or Butterworth).
2. The golden matrix is synthesized automatically.
3. Edit → change individual Mij or Ri values.
4. Watch S11 / S21 update live.

GOLDEN vs CURRENT
• M_golden = immutable reference design.
• M_current = working matrix you tune.
• Reset restores golden. Undo steps back.

TRANSMISSION ZEROS
• TZ → Design TZs: enter zero frequencies in MHz, comma separated (up to n−2, each outside the passband).
• The golden matrix gets cross-couplings (e.g. M1_5, M2_4); tune them in Edit.
• TZ Info lists the zeros of the current matrix next to the design.
• Clear TZs returns to the all-pole prototype. Dotted pink lines mark designed zeros.

ANALYSIS
• Diagnostics – full self-test + extracted Qe/k.
• Poles – approximate natural frequencies + GD variation.
• Sensitivity – ±5 % perturbation of sequential couplings.

FILES
• Save s2p exports the current response; Load s2p overlays a saved or measured file.
• De-embed delay (ps) adjusts reference plane.

MARKERS & PLOT
• Set up to 3 markers; deltas appear automatically.
• Golden overlay shows ideal response.
• Zoom Band sets sweep ≈ 3× design BW.
• Ch1 cycles S11 → S22 → Smith → none. Smith plots S11 on a Smith chart
  (Ch2 is hidden in this mode); marker boxes also show Γ and Z (50 Ω).

TIPS
• Keep Qu realistic (1000–20000 typical).
• High condition-number HUD means matrix is extreme."""


# ===================== SAFE DIALOGS (Pythonista cancel → KeyboardInterrupt) =====================

def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (KeyboardInterrupt, Exception):
        return None


def alert(title, message='', *a, **k):
    return _safe(dialogs.alert, title, message, *a, **k)


def confirm(title, message='', yes='Yes', no='No'):
    """True only if the first button was tapped; cancel -> False."""
    return _safe(dialogs.alert, title, message, yes, no) == 1


def hud(message):
    _safe(dialogs.hud_alert, message)


def ask(title, message='', *a, **k):
    return _safe(dialogs.input_alert, title, message, *a, **k)


def pick(title, items):
    return _safe(dialogs.list_dialog, title, items)


def form(title, fields):
    return _safe(dialogs.form_dialog, title, fields)


def show_text(title, text):
    _safe(dialogs.text_dialog, title, text)


# ===================== MODEL =====================

class Marker:
    def __init__(self, color='red'):
        self.freq = None
        self.color = color
        self.enabled = True
        self.value = self.unit = self.trace = self.gamma = None

    def update(self, sim, state):
        self.gamma = None
        if self.freq is None:
            return
        f = sim.freqs
        if not (f[0] <= self.freq <= f[-1]):
            self.value = None
            return
        if state.ch1_mode != 'none':
            is11 = state.ch1_mode in ('s11', 'smith')
            self.value = np.interp(self.freq, f, sim.s11_db if is11 else sim.s22_db)
            self.unit, self.trace = 'dB', 'S11' if is11 else 'S22'
            if state.ch1_mode == 'smith':
                self.gamma = complex(np.interp(self.freq, f, sim.s11.real),
                                     np.interp(self.freq, f, sim.s11.imag))
        elif state.secondary_plot == 'none':
            self.value = self.unit = self.trace = None
        else:
            cfg = PLOT_MODES.get(state.secondary_plot, PLOT_MODES['loss'])
            self.value = np.interp(self.freq, f, getattr(sim, cfg['key']))
            self.unit = cfg['unit']
            self.trace = f"S21 ({state.secondary_plot})"


def marker_deltas(markers):
    active = [(i + 1, m) for i, m in enumerate(markers)
              if m.enabled and m.freq is not None and m.value is not None]
    return [(i1, i2, (m2.freq - m1.freq) / 1e6, m2.value - m1.value, m1.unit or '')
            for a, (i1, m1) in enumerate(active) for (i2, m2) in active[a + 1:]]


class CATState:
    def __init__(self):
        self.n = 5
        self.f0 = 500e6
        self.bw_design = 50e6
        self.fbw_design = self.bw_design / self.f0
        self.qu = 500.0
        self.Qe_source = self.Qe_load = None
        self.start_freq, self.stop_freq = 350e6, 650e6
        self.num_points = DEFAULT_NUM_POINTS
        self.ripple_dB = 0.02
        self.prototype = 'chebyshev'            # 'chebyshev' | 'butterworth'
        self.rl_ymin, self.rl_ymax = -80.0, 5.0
        self.il_ymin, self.il_ymax = -100.0, 5.0
        self.phase_ymin, self.phase_ymax = -180.0, 180.0
        self.gd_ymin = self.gd_ymax = None
        self.secondary_plot = 'loss'
        self.ch1_mode = 's11'                    # 's11' | 's22' | 'smith' | 'none'
        self.phase_flatten = False
        self.phase_delay = self.phase_constant = 0.0
        self.deembed_delay = 0.0
        self.tz_freqs = [435e6 , 565e6]                       # designed transmission zeros (Hz)
        self.show_golden_ref = False
        self.show_meas = False
        self.DOCS = os.path.join(os.path.expanduser('~/Documents'), 'S2P')
        os.makedirs(self.DOCS, exist_ok=True)
        self.loaded_freqs = self.loaded_s11 = self.loaded_s21 = self.loaded_s22 = None
        self.loaded_s2p_path = None
        self.g = self.M_golden = self.M_current = None
        self.markers = [Marker(c) for c in ('red', '#0044ff', '#aa00ff')]
        self.undo_stack = deque(maxlen=MAX_UNDO)
        self.last_sim = None
        self.sim_dirty = True

    def push_undo(self):
        if self.M_current is not None:
            self.undo_stack.append(self.M_current.copy())

    def undo(self):
        if not self.undo_stack:
            return False
        self.M_current = self.undo_stack.pop()
        self.sim_dirty = True
        return True

    def mark_dirty(self):
        self.sim_dirty = True
        self.last_sim = None

    def clear_meas(self):
        self.loaded_freqs = self.loaded_s11 = self.loaded_s21 = self.loaded_s22 = None
        self.loaded_s2p_path = None
        self.show_meas = False


# ===================== PROTOTYPE SYNTHESIS =====================

def chebyshev_g_values(n, ripple_db):
    if n < 2:
        raise ValueError('n must be >= 2')
    if ripple_db < 0:
        raise ValueError('Ripple must be >= 0 dB')
    eps = np.sqrt(10**(ripple_db / 10) - 1)
    if eps <= 1e-15:
        raise ValueError('Ripple must be > 0 for Chebyshev.')
    s = np.sinh(np.arcsinh(1 / eps) / n)
    a = [None] + [np.sin((2 * i - 1) * np.pi / (2 * n)) for i in range(1, n + 1)]
    b = [None] + [s * s + np.sin(i * np.pi / n)**2 for i in range(1, n + 1)]
    g = [1.0, 2 * a[1] / s]
    for i in range(2, n + 1):
        g.append((4 * a[i - 1] * a[i]) / (b[i - 1] * g[i - 1]))
    g.append(g[1] / g[n])
    return g


def butterworth_g_values(n):
    if n < 2:
        raise ValueError('n must be >= 2')
    return [1.0] + [2.0 * np.sin((2 * i - 1) * np.pi / (2 * n)) for i in range(1, n + 1)] + [1.0]


def build_matrix(state):
    n = state.n
    M = np.zeros((n + 2, n + 2))
    if state.Qe_source is not None:
        M[0, 1] = M[1, 0] = 1.0 / np.sqrt(state.Qe_source * state.fbw_design)
    if state.Qe_load is not None:
        M[n, n + 1] = M[n + 1, n] = 1.0 / np.sqrt(state.Qe_load * state.fbw_design)
    for i in range(1, n):
        M[i, i + 1] = M[i + 1, i] = 1 / np.sqrt(state.g[i] * state.g[i + 1])
    return M


# ---------- transmission zeros (normalized lowpass variable Ω) ----------

def freq_to_omega(f, f0, fbw):
    return (f / f0 - f0 / f) / fbw


def omega_to_freq(w, f0, fbw):
    x = w * fbw
    return f0 * (x + np.sqrt(x * x + 4.0)) / 2.0


def parse_tz_text(text):
    """'470, 535.5' (MHz) -> [470e6, 535.5e6]"""
    return [float(x) * 1e6 for x in str(text).replace(';', ',').split(',') if x.strip()]


def check_tzs(n, freqs, f0, fbw):
    """Error message for an unusable zero list, else None."""
    if not freqs:
        return None
    if len(freqs) > n - 2:
        return f'n = {n} allows at most {max(n - 2, 0)} finite transmission zeros.'
    if any(f <= 0 for f in freqs):
        return 'Zero frequencies must be positive.'
    om = [freq_to_omega(f, f0, fbw) for f in freqs]
    if any(abs(w) < 1.05 for w in om):
        return 'Each zero must lie outside the passband (|Ω| > 1.05).'
    if len(set(np.round(om, 6))) != len(om):
        return 'Zeros must be distinct.'
    return None


def tz_polynomials(n, tz, ripple_dB):
    """Filtering polynomials F (reflection zeros) and P' (transmission) for n poles and
    finite zeros `tz` (normalized Ω), via the Chebyshev-type recursion."""
    W2 = np.array([1.0, 0.0, -1.0])                      # Ω2 − 1
    U, V = np.array([1.0]), np.array([0.0])
    D = np.array([1.0])
    for w in list(tz) + [np.inf] * (n - len(tz)):
        a = 0.0 if np.isinf(w) else 1.0 / w
        c = np.sqrt(1.0 - a * a)
        lin = np.array([1.0, -a])
        U, V = (np.polyadd(np.polymul(U, lin), c * np.polymul(W2, V)),
                np.polyadd(np.polymul(V, lin), c * U))
        if not np.isinf(w):
            D = np.polymul(D, np.array([-a, 1.0]))      # (1 − Ω/w)
    eps = np.sqrt(10 ** (ripple_dB / 10) - 1)
    return U, D / eps


def tz_coupling_matrix(n, tz, ripple_dB):
    """(n+2)×(n+2) coupling matrix with transmission zeros at `tz` (normalized Ω) and
    equiripple passband: polynomials -> eigen-decomposition of the admittance -> two-ended
    Gram-Schmidt reduction to the folded form (cross-couplings on the two anti-diagonals)."""
    F, P = tz_polynomials(n, tz, ripple_dB)
    roots = np.roots(np.polyadd(np.polymul(F, F), np.polymul(P, P)))
    upper = roots[roots.imag > 0]
    if len(upper) != n:
        raise ValueError('Transmission-zero synthesis failed (polynomial roots).')
    E = F[0] * np.poly(upper)
    Dn = np.polyadd(E.real, F)                           # poles of y21 (eigenvalues)
    lam = np.sort(np.roots(Dn).real)
    dD = np.polyval(np.polyder(Dn), lam)
    s2 = -np.polyval(E.imag, lam) / dD
    s = np.sqrt(np.clip(s2, 0.0, None))
    if np.any(s < 1e-12):
        raise ValueError('Transmission-zero synthesis failed (zero residue).')
    l = np.polyval(P, lam) / dD / s
    T = np.zeros((n, n))
    T[:, 0] = s / np.linalg.norm(s)
    T[:, n - 1] = l / np.linalg.norm(l)
    known = [0, n - 1]

    def next_vec(src, dst):
        K = T[:, known]
        w = lam * T[:, src]
        for _ in range(2):
            w = w - K @ (K.T @ w)
        nw = np.linalg.norm(w)
        if nw < 1e-10:
            raise ValueError('Transmission-zero synthesis failed (degenerate).')
        T[:, dst] = w / nw
        known.append(dst)

    f, b = 0, n - 1
    while b - f > 1:
        next_vec(f, f + 1)
        f += 1
        if b - f > 1:
            next_vec(b, b - 1)
            b -= 1
    M = np.zeros((n + 2, n + 2))
    M[1:n + 1, 1:n + 1] = T.T @ (lam[:, None] * T)
    M[0, 1] = M[1, 0] = float(s @ T[:, 0])
    M[n, n + 1] = M[n + 1, n] = float(l @ T[:, n - 1])
    return M


def matrix_tz_omegas(n, M):
    """Finite transmission zeros (complex, normalized Ω) of a coupling matrix."""
    M = np.asarray(M, dtype=float)
    s, l = M[0, 1:n + 1], M[n + 1, 1:n + 1]
    lam, V = np.linalg.eigh(M[1:n + 1, 1:n + 1])
    c = (V.T @ s) * (V.T @ l)
    q = np.zeros(1)
    for k in range(n):
        q = np.polyadd(q, c[k] * np.poly(np.delete(lam, k)))
    scale = max(np.linalg.norm(s) * np.linalg.norm(l), 1e-12)
    while len(q) > 1 and abs(q[0]) < 1e-7 * scale:
        q = q[1:]
    return np.roots(q) if len(q) > 1 else np.array([])


def synthesize(state):
    """Golden matrix from the prototype (or from the transmission zeros); resets M_current."""
    if state.tz_freqs:
        om = [freq_to_omega(f, state.f0, state.fbw_design) for f in state.tz_freqs]
        state.M_golden = tz_coupling_matrix(state.n, om, state.ripple_dB)
        state.g = chebyshev_g_values(state.n, state.ripple_dB)      # kept for validation only
        state.Qe_source = 1.0 / (state.M_golden[0, 1]**2 * state.fbw_design)
        state.Qe_load = 1.0 / (state.M_golden[state.n, state.n + 1]**2 * state.fbw_design)
    else:
        if state.prototype == 'butterworth':
            state.g = butterworth_g_values(state.n)
        else:
            state.g = chebyshev_g_values(state.n, state.ripple_dB)
        state.Qe_source = state.g[0] * state.g[1] / state.fbw_design
        state.Qe_load = state.g[state.n] * state.g[state.n + 1] / state.fbw_design
        state.M_golden = build_matrix(state)
    state.M_current = state.M_golden.copy()


def cross_pairs(n):
    """Cross-coupling positions of the folded form (anti-diagonals i+j = n+1 and n+2)."""
    return [(i, j) for i in range(1, n + 1) for j in range(i + 2, n + 1) if i + j in (n + 1, n + 2)]


def build_labels(n):
    labels = ['M_S1']
    for i in range(1, n):
        labels += [f'R{i}', f'M{i}_{i+1}']
    labels += [f'R{n}', f'M{n}L']
    return labels + [f'M{i}_{j}' for i, j in cross_pairs(n)]


def label_to_ij(n, lbl):
    if lbl == 'M_S1':
        return 0, 1
    if lbl == f'M{n}L':
        return n, n + 1
    if lbl.startswith('M'):
        a, b = lbl[1:].split('_', 1)
        return int(a), int(b)
    k = int(lbl[1:])
    return k, k


# ===================== VALIDATION =====================

def validate_matrix(state, M, name='Matrix'):
    if M is None:
        return [f'{name} is None.']
    expected = (state.n + 2, state.n + 2)
    if M.shape != expected:
        return [f'{name} shape {M.shape}; expected {expected}.']
    errors = []
    if not np.all(np.isfinite(M)):
        errors.append(f'{name} contains NaN or Inf.')
    if not np.allclose(M, M.T, atol=1e-10):
        errors.append(f'{name} is not symmetric.')
    return errors


def validate_design(state):
    e = []
    if state.n < 2:
        e.append('Filter order n must be >= 2.')
    if state.f0 <= 0:
        e.append('Center frequency must be > 0.')
    if state.bw_design <= 0:
        e.append('Design bandwidth must be > 0.')
    if state.fbw_design <= 0 or state.fbw_design > MAX_FBW:
        e.append(f'Fractional bandwidth must be in ({MIN_FBW}, {MAX_FBW}].')
    if state.qu < MIN_QU or state.qu > MAX_QU:
        e.append(f'Unloaded Q must be in [{MIN_QU}, {MAX_QU}].')
    if state.num_points < 2:
        e.append('Number of points must be >= 2.')
    if state.start_freq <= 0 or state.stop_freq <= state.start_freq:
        e.append('Frequency sweep is invalid.')
    if state.g is None:
        e.append('Prototype g-values are missing.')
    elif len(state.g) != state.n + 2:
        e.append(f'g has {len(state.g)} values; expected {state.n + 2}.')
    elif not np.all(np.isfinite(state.g)):
        e.append('g-values contain NaN or Inf.')
    e += validate_matrix(state, state.M_golden, 'M_golden')
    e += validate_matrix(state, state.M_current, 'M_current')
    if state.Qe_source is None or state.Qe_source <= 0:
        e.append('Invalid source external Q.')
    if state.Qe_load is None or state.Qe_load <= 0:
        e.append('Invalid load external Q.')
    return e


# ===================== SIGNAL PROCESSING =====================

def to_db(s):
    return 20 * np.log10(np.abs(s) + 1e-20)


def unwrap_phase(s):
    return np.unwrap(np.angle(s))


def estimate_linear_delay(freqs, phase_unw):
    if len(freqs) < 2:
        return 0.0
    return -np.polyfit(freqs, phase_unw, 1)[0] / (2 * np.pi)


def flatten_phase(freqs, phase_deg, delay, constant=0.0):
    return phase_deg + 360 * freqs * delay + constant


def deembed(freqs, s, extra_delay=0.0, auto=False):
    """Shift a reflection trace's reference plane by extra_delay (s).
    auto=True also removes the fitted linear phase (for raw measurements with cable delay)."""
    delay = extra_delay
    if auto:
        delay += estimate_linear_delay(freqs, unwrap_phase(s))
    return s * np.exp(2j * np.pi * freqs * delay)


def compute_phase_and_gd(freqs, s21):
    phase_unw = unwrap_phase(s21)
    gd = np.zeros_like(freqs)
    if len(freqs) > 1:
        gd[1:] = -np.diff(phase_unw) / np.diff(2 * np.pi * freqs)
        gd[0] = gd[1]
    return phase_unw / np.pi * 180, gd * 1e9


def linear_interp1d(x, y):
    """np.interp with linear extrapolation outside the data range."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return lambda xn: np.full(np.shape(xn), np.nan)
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    s0 = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
    s1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0

    def f(xn):
        xn = np.asarray(xn, dtype=float)
        out = np.interp(xn, xs, ys)
        out = np.where(xn < xs[0], ys[0] + s0 * (xn - xs[0]), out)
        return np.where(xn > xs[-1], ys[-1] + s1 * (xn - xs[-1]), out)
    return f


def prepare_meas(state, sim_freqs):
    """Interpolate the loaded .s2p onto the simulation grid for plotting."""
    if state.loaded_freqs is None or not state.show_meas:
        return None
    lf = state.loaded_freqs
    overlap_ok = True
    if len(lf) > 1:
        overlap = max(0.0, min(lf[-1], sim_freqs[-1]) - max(lf[0], sim_freqs[0]))
        span = sim_freqs[-1] - sim_freqs[0]
        overlap_ok = not (span > 0 and overlap / span < 0.3)
    return_db = return_gamma = None
    if state.ch1_mode in ('s11', 'smith') and state.loaded_s11 is not None:
        return_db = linear_interp1d(lf, to_db(state.loaded_s11))(sim_freqs)
        if state.ch1_mode == 'smith':
            inside = (sim_freqs >= lf[0]) & (sim_freqs <= lf[-1])      # no extrapolation on a Smith chart
            gr = np.interp(sim_freqs, lf, state.loaded_s11.real)
            gi = np.interp(sim_freqs, lf, state.loaded_s11.imag)
            return_gamma = np.where(inside, gr + 1j * gi, np.nan)
    elif state.ch1_mode == 's22' and state.loaded_s22 is not None:
        return_db = linear_interp1d(lf, to_db(state.loaded_s22))(sim_freqs)
    mode, s21_plot = state.secondary_plot, None
    if mode == 'loss':
        s21_plot = linear_interp1d(lf, to_db(state.loaded_s21))(sim_freqs)
    elif mode == 'phase':
        ph = linear_interp1d(lf, unwrap_phase(state.loaded_s21))(sim_freqs) / np.pi * 180
        s21_plot = ph + 360 * sim_freqs * state.phase_delay + state.phase_constant
    elif mode == 'gd':
        s21_plot = linear_interp1d(lf, compute_phase_and_gd(lf, state.loaded_s21)[1])(sim_freqs)
    return SimpleNamespace(s21_plot=s21_plot, return_db=return_db,
                           return_gamma=return_gamma, overlap_ok=overlap_ok)


# ===================== SYSTEM MATRIX & S-PARAMS =====================

def build_system_matrix(state, M, Omega):
    n, N = state.n, state.n + 2
    U = np.zeros((N, N))
    U[1:n + 1, 1:n + 1] = np.eye(n)
    R = np.zeros((N, N))
    R[range(1, n + 1), range(1, n + 1)] = 1.0 / (state.qu * state.fbw_design)
    G = np.zeros((N, N))
    G[0, 0] = G[N - 1, N - 1] = 1.0
    Mmat = np.asarray(M, dtype=float)
    A = np.zeros((len(Omega), N, N), dtype=complex)
    for k, om in enumerate(Omega):
        A[k] = G + R + 1j * (om * U - Mmat)
    return A


def solve_excitation(A, port):
    nf, N, _ = A.shape
    b = np.zeros((nf, N), dtype=complex)
    b[:, port] = 1.0
    try:
        return np.linalg.solve(A, b[..., np.newaxis])[..., 0]
    except Exception:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros((nf, N), dtype=complex)


def compute_s_safe(freqs, M, state):
    freqs = np.asarray(freqs, dtype=float)
    if np.any(freqs <= 0):
        raise ValueError('Frequency must be > 0.')
    Omega = (freqs / state.f0 - state.f0 / freqs) / state.fbw_design
    A = build_system_matrix(state, M, Omega)
    cond_max = 0.0
    for idx in (0, len(freqs) // 2, -1):
        try:
            c = np.linalg.cond(A[idx])
            if np.isfinite(c):
                cond_max = max(cond_max, c)
        except Exception:
            pass
    if cond_max > COND_FAIL:
        raise ValueError(f'System matrix severely ill-conditioned (cond ≈ {cond_max:.2e}).')
    x1 = solve_excitation(A, 0)
    x2 = solve_excitation(A, -1)
    return (1 - 2 * x1[:, 0], -2 * x1[:, -1], -2 * x2[:, 0], 1 - 2 * x2[:, -1], cond_max)


def compute_s(freqs, M, state):
    return compute_s_safe(freqs, M, state)[:4]


def simulate(state):
    if not state.sim_dirty and state.last_sim is not None:
        return state.last_sim
    errors = validate_design(state)
    if errors:
        raise ValueError('Invalid filter state:\n' + '\n'.join(errors))
    if state.fbw_design < MIN_FBW:
        raise ValueError('FBW out of safe range')
    freqs = np.linspace(state.start_freq, state.stop_freq, state.num_points)
    s11, s21, s12, s22, cond_max = compute_s_safe(freqs, state.M_current, state)
    phase_deg, gd_ns = compute_phase_and_gd(freqs, s21)
    phase_rel = (flatten_phase(freqs, phase_deg, state.phase_delay, state.phase_constant)
                 if state.phase_flatten else phase_deg)
    sim = SimpleNamespace(
        freqs=freqs, s11=s11, s21=s21, s22=s22,
        s11_db=to_db(s11), s21_db=to_db(s21), s22_db=to_db(s22),
        phase_deg=phase_deg, gd_ns=gd_ns, phase_rel=phase_rel,
        energy_error=np.max(np.abs(np.abs(s11)**2 + np.abs(s21)**2 - 1.0)),
        cond_max=cond_max)
    state.last_sim = sim
    state.sim_dirty = False
    return sim


def golden_traces(state, sim):
    """(complex S11, secondary-plot data) of the golden matrix on the simulation grid."""
    try:
        g11, g21, _, _, _ = compute_s_safe(sim.freqs, state.M_golden, state)
        mode = state.secondary_plot
        data = None
        if mode == 'loss':
            data = to_db(g21)
        elif mode in ('phase', 'gd'):
            phase, gd = compute_phase_and_gd(sim.freqs, g21)
            if mode == 'gd':
                data = gd
            else:
                data = (flatten_phase(sim.freqs, phase, state.phase_delay, state.phase_constant)
                        if state.phase_flatten else phase)
        return g11, data
    except Exception:
        return None, None


# ===================== ANALYSIS =====================

def compute_poles_approx(state):
    """Approximate natural frequencies (Hz, complex) from the resonator sub-block."""
    if state.M_current is None:
        return None
    n = state.n
    Msub = state.M_current[1:n + 1, 1:n + 1] + 1j / (state.qu * state.fbw_design) * np.eye(n)
    try:
        k = 0.5 * state.fbw_design * state.f0
        poles = [complex(state.f0 + k * e.real, k * e.imag) for e in np.linalg.eigvals(Msub)]
        return sorted(poles, key=lambda p: p.real)
    except Exception:
        return None


def group_delay_variation(state, sim):
    mask = (sim.freqs >= state.f0 - state.bw_design / 2) & (sim.freqs <= state.f0 + state.bw_design / 2)
    if not np.any(mask):
        return None
    return float(np.nanmax(sim.gd_ns[mask]) - np.nanmin(sim.gd_ns[mask]))


def run_sensitivity(state, perturb=SENS_PERTURB):
    """Perturb each coupling by ±perturb; report max |ΔS11| / |ΔS21| (dB)."""
    if state.M_current is None:
        return 'No matrix'
    n = state.n
    freqs = np.linspace(state.start_freq, state.stop_freq, 101)
    s11, s21, _, _, _ = compute_s_safe(freqs, state.M_current, state)
    base11, base21 = to_db(s11), to_db(s21)
    pairs = [(i, i + 1) for i in range(1, n)]
    pairs += [(i, j) for i, j in cross_pairs(n) if abs(state.M_current[i, j]) > 1e-9]
    lines = [f'Sensitivity (±{perturb*100:.0f}% on couplings)\n']
    for i, j in pairs:
        d11s, d21s = [], []
        for sign in (+1, -1):
            M = state.M_current.copy()
            step = perturb * abs(M[i, j]) if abs(M[i, j]) > 1e-9 else perturb * 0.05
            M[i, j] += sign * step
            M[j, i] = M[i, j]
            try:
                p11, p21, _, _, _ = compute_s_safe(freqs, M, state)
                d11s.append(np.nanmax(np.abs(to_db(p11) - base11)))
                d21s.append(np.nanmax(np.abs(to_db(p21) - base21)))
            except Exception:
                d11s.append(np.nan)
                d21s.append(np.nan)
        lines.append(f'M{i}_{j}:  max|ΔS11|={np.nanmax(d11s):.2f} dB   max|ΔS21|={np.nanmax(d21s):.2f} dB')
    return '\n'.join(lines)


# ===================== DIAGNOSTICS =====================

def _make_state(**overrides):
    st = CATState()
    for k, v in overrides.items():
        setattr(st, k, v)
    synthesize(st)
    return st


def report_extracted_parameters(state):
    if state.M_current is None:
        return 'No matrix available.'
    M, n, fbw = state.M_current, state.n, state.fbw_design

    def qe(m):
        return 1.0 / (m**2 * fbw) if m > 1e-12 else float('inf')
    lines = [f'Extracted Qe_source ≈ {qe(abs(M[0, 1])):.1f}',
             f'Extracted Qe_load   ≈ {qe(abs(M[n, n + 1])):.1f}']
    if state.Qe_source:
        lines.append(f'  (design Qe_s = {state.Qe_source:.1f})')
    if state.Qe_load:
        lines.append(f'  (design Qe_l = {state.Qe_load:.1f})')
    lines.append('Sequential couplings:')
    for i in range(1, n):
        lines.append(f'  M{i}_{i+1} = {M[i, i+1]:.6f}')
        if state.M_golden is not None:
            lines.append(f'    Δ vs golden = {M[i, i+1] - state.M_golden[i, i+1]:+.6f}')
    for i, j in cross_pairs(n):
        if abs(M[i, j]) > 1e-9 or (state.M_golden is not None and abs(state.M_golden[i, j]) > 1e-9):
            lines.append(f'  M{i}_{j} (cross) = {M[i, j]:.6f}')
    return '\n'.join(lines)


def run_all_diagnostics(state):
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # synthesis
    for n in range(2, 7):
        for ripple in (0.01, 0.1, 0.5):
            try:
                g = chebyshev_g_values(n, ripple)
                check(len(g) == n + 2 and np.all(np.isfinite(g)) and np.all(np.asarray(g[1:n + 1]) > 0),
                      f'Chebyshev g invalid for n={n}, ripple={ripple}.')
            except Exception as e:
                fails.append(f'Chebyshev failed n={n}: {e}')
    # reference filter: matrix properties + RF behaviour
    ref = _make_state()
    M = ref.M_golden
    check(M.shape == (ref.n + 2, ref.n + 2) and np.allclose(M, M.T, atol=1e-12) and np.all(np.isfinite(M)),
          'Golden matrix invalid.')
    freqs = np.linspace(ref.start_freq, ref.stop_freq, ref.num_points)
    try:
        s11, s21, s12, s22, _ = compute_s_safe(freqs, ref.M_current, ref)
        for name, arr in (('S11', s11), ('S21', s21), ('S12', s12), ('S22', s22)):
            check(np.all(np.isfinite(arr)), f'{name} contains NaN/Inf.')
        err = np.max(np.abs(s21 - s12))
        check(err < 1e-8, f'Reciprocity failed: {err:.3e}.')
        err = np.max(np.abs(np.abs(s11) - np.abs(s22)))
        check(err < 1e-8, f'Port symmetry failed: {err:.3e}.')
        lossless = _make_state(qu=1e12)
        l11, l21, _, _, _ = compute_s_safe(freqs, lossless.M_current, lossless)
        err = np.max(np.abs(np.abs(l11)**2 + np.abs(l21)**2 - 1.0))
        check(err < 1e-6, f'Lossless energy failed: {err:.3e}.')
    except Exception as e:
        fails.append(f'Simulation failed: {e}')
    # current design passband
    try:
        pf = np.linspace(state.f0 - state.bw_design / 2, state.f0 + state.bw_design / 2, 201)
        p11, p21, _, _, _ = compute_s_safe(pf, state.M_current, state)
        rl, il = to_db(p11), to_db(p21)
        check(np.max(rl) <= -10.0, f'Passband RL weak: max |S11| ≈ {np.max(rl):.1f} dB')
        check(np.min(il) >= -3.0, f'Passband IL high: min |S21| ≈ {np.min(il):.1f} dB')
    except Exception as e:
        fails.append(f'Passband check failed: {e}')

    # transmission-zero synthesis
    try:
        tzf = [470e6, 535e6]
        tzs = _make_state(n=6, qu=1e12, tz_freqs=tzf)
        z21 = compute_s_safe(np.array(tzf), tzs.M_current, tzs)[1]
        check(np.max(np.abs(z21)) < 1e-4, 'TZ synthesis: S21 not zero at the designed zeros.')
        pf = omega_to_freq(np.linspace(-1.0, 1.0, 201), tzs.f0, tzs.fbw_design)
        check(np.max(to_db(compute_s_safe(pf, tzs.M_current, tzs)[0])) < -20.0,
              'TZ synthesis: passband return loss too weak.')
    except Exception as e:
        fails.append(f'TZ synthesis failed: {e}')

    extracted = report_extracted_parameters(state)
    if not fails:
        report = ('RF DIAGNOSTICS: PASS\n\n✓ Chebyshev synthesis\n✓ Golden matrix properties\n'
                  '✓ RF simulation (finite, reciprocal, symmetric, lossless energy)\n'
                  '✓ Passband RL/IL sanity\n✓ Transmission-zero synthesis\n\n--- Extracted from M_current ---\n' + extracted)
        if state.last_sim is not None:
            report += (f'\n\nMax sampled condition ≈ {state.last_sim.cond_max:.2e}'
                       f'\nEnergy residual ≈ {state.last_sim.energy_error:.3e}')
        return report
    return (f'RF DIAGNOSTICS: FAIL\n\n{len(fails)} issue(s):\n\n'
            + ''.join(f'{i}. {f}\n' for i, f in enumerate(fails, 1))
            + '\n--- Extracted ---\n' + extracted)


# ===================== S2P I/O =====================

def write_s2p(path, freqs, s11, s21, s12, s22, z0=50.0):
    with open(path, 'w') as f:
        f.write('! Exported by Pythonista CAT Tool\n')
        f.write(f'# Hz S RI R {z0:.6g}\n')
        for fr, a11, a21, a12, a22 in zip(freqs, s11, s21, s12, s22):
            f.write(f'{fr:.0f} ' + ' '.join(f'{v.real:.6e} {v.imag:.6e}' for v in (a11, a21, a12, a22)) + '\n')


def read_s2p(path):
    """Returns (freqs, s11, s21, s22) sorted by frequency, or four Nones on failure."""
    mult = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    unit, fmt, rows = 'MHZ', 'RI', []

    def cplx(a, b):
        a, b = float(a), float(b)
        if fmt == 'RI':
            return complex(a, b)
        if fmt == 'MA':
            return a * np.exp(1j * b * np.pi / 180)
        if fmt == 'DB':
            return 10**(a / 20) * np.exp(1j * b * np.pi / 180)
        raise ValueError(f'Unsupported format: {fmt}')
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                if line.startswith('#'):
                    p = line[1:].upper().split()
                    if len(p) >= 5:
                        unit, fmt = p[0], p[2]
                    continue
                p = line.split()
                if len(p) != 9:
                    continue
                rows.append([float(p[0]) * mult.get(unit, 1e6)]
                            + [cplx(p[k], p[k + 1]) for k in (1, 3, 5, 7)])   # S11 S21 S12 S22
        if not rows:
            return None, None, None, None
        rows.sort(key=lambda r: r[0])
        col = lambda k: np.array([r[k] for r in rows])
        return col(0), col(1), col(2), col(4)
    except Exception as e:
        print(f'Error reading s2p: {e}')
        return None, None, None, None


# ===================== PLOT =====================

_FIG = {}


def _figure():
    """One figure for the whole session (re-creating it every redraw exhausts memory)."""
    if 'fig' not in _FIG:
        _FIG['fig'] = plt.figure(figsize=(10, 7.5), facecolor='#1a1a1a')
    fig = _FIG['fig']
    fig.clf()
    return fig


def render_plot(state, sim, meas, golden, deltas):
    """Draw the dual-axis figure (or the Smith chart) and return PNG bytes."""
    g11, g21_data = golden
    if state.ch1_mode == 'smith':
        return render_smith(state, sim, meas, g11, deltas)
    if True:
        fig = _figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        ax1.set_xlabel('Frequency (MHz)')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        x = sim.freqs / 1e6

        if state.ch1_mode != 'none':
            ax1.set_ylabel('Return Loss (dB)', color='#00ddff')
            ax1.tick_params(axis='y', colors='#00ddff')
            ax1.set_ylim(state.rl_ymin, state.rl_ymax)
            ch1 = sim.s11_db if state.ch1_mode == 's11' else sim.s22_db
            ax1.plot(x, ch1, '#00ddff', lw=2.2, label='S11 Sim')
            if g11 is not None and state.ch1_mode == 's11':
                ax1.plot(x, to_db(g11), '#00ddff', lw=1.2, linestyle=':', alpha=0.7, label='S11 Golden')
            if meas and meas.return_db is not None:
                ax1.plot(x, meas.return_db, '#ffaa00', lw=1.8, label='S11 Meas', linestyle='--')

        mode = state.secondary_plot
        if mode != 'none':
            cfg = PLOT_MODES.get(mode, PLOT_MODES['loss'])
            ax2.set_ylabel(cfg['ylabel'], color=cfg['color_sim'])
            ax2.tick_params(axis='y', colors=cfg['color_sim'])
            if mode == 'loss':
                ax2.set_ylim(state.il_ymin, state.il_ymax)
            elif mode == 'phase':
                ax2.set_ylim(state.phase_ymin, state.phase_ymax)
            elif mode == 'gd' and state.gd_ymin is not None:
                ax2.set_ylim(state.gd_ymin, state.gd_ymax)
            ax2.plot(x, getattr(sim, cfg['key']), cfg['color_sim'], lw=2.2,
                     label=f'Sim ({cfg["name"]})')
            if g21_data is not None:
                ax2.plot(x, g21_data, cfg['color_sim'], lw=2.2, linestyle=':',
                         label=f'Golden ({cfg["name"]})')
            if meas and meas.s21_plot is not None:
                ax2.plot(x, meas.s21_plot, cfg['color_meas'], lw=2.2, linestyle='--',
                         label=f'Meas ({cfg["name"]})')

        # markers
        marker_ax = ax1 if state.ch1_mode != 'none' else ax2
        info = []
        for i, m in enumerate(state.markers):
            if not m.enabled or m.freq is None or m.value is None:
                continue
            if not (sim.freqs[0] <= m.freq <= sim.freqs[-1]):
                continue
            mx = m.freq / 1e6
            marker_ax.axvline(mx, color=m.color, linestyle='--', linewidth=1.0, alpha=0.7)
            marker_ax.plot(mx, m.value, marker='o', color=m.color, markersize=7,
                           markeredgecolor='white', markeredgewidth=1.5)
            info.append((i + 1, mx, m.value, m.unit, m.trace))
        if info or deltas:
            lines = ['MARKERS']
            for num, fr, val, unit, trace in info:
                lines += [f'M{num}: {fr:.3f} MHz', f'   {trace} = {val:.3f}{unit}']
            if deltas:
                lines.append('--- Δ ---')
                for i1, i2, df, dv, unit in deltas:
                    lines += [f'M{i1}-M{i2}: {df:+.3f} MHz', f'   Δ = {dv:+.3f}{unit}']
            ax1.text(0.98, 0.98, '\n'.join(lines), transform=ax1.transAxes, color='white',
                     fontsize=9, fontfamily='monospace', verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117',
                               edgecolor='#555555', alpha=0.95))

        first = True
        for tz in state.tz_freqs:
            if sim.freqs[0] <= tz <= sim.freqs[-1]:
                ax1.axvline(tz / 1e6, color='#ff6688', linestyle=':', linewidth=1.2, alpha=0.8,
                            label='TZ (design)' if first else None)
                first = False

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(h1 + h2, l1 + l2, loc='upper left')

        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, facecolor='#1a1a1a')
        fig.clf()
        return buf.getvalue()


def _smith_grid(ax):
    t = np.linspace(0, 2 * np.pi, 721)
    grid = dict(color='#666666', lw=0.7)
    txt = dict(color='#999999', fontsize=7, ha='center', va='center')
    ax.plot(np.cos(t), np.sin(t), color='#aaaaaa', lw=1.4)
    ax.plot([-1, 1], [0, 0], **grid)
    for r in (0.2, 0.5, 1.0, 2.0, 5.0):                      # constant-resistance circles
        ax.plot((r + np.cos(t)) / (1 + r), np.sin(t) / (1 + r), **grid)
        ax.text((r - 1) / (r + 1) + 0.045, 0.035, f'{r:g}', **txt)
    for xr in (0.2, 0.5, 1.0, 2.0, 5.0):                     # constant-reactance arcs
        for sgn in (1, -1):
            rad = 1.0 / xr
            gx = 1 + rad * np.cos(t)
            gy = sgn * rad + rad * np.sin(t)
            bad = gx * gx + gy * gy > 1.0005
            ax.plot(np.where(bad, np.nan, gx), np.where(bad, np.nan, gy), **grid)
            d = xr * xr + 1
            ax.text(1.08 * (xr * xr - 1) / d, 1.08 * sgn * 2 * xr / d, f'{sgn * xr:+g}', **txt)


def render_smith(state, sim, meas, g11, deltas):
    """S11 on a Smith chart (normalized to the 50 Ω reference); returns PNG bytes."""
    fig = _figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#1a1a1a')
    ax.set_aspect('equal')
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis('off')
    ax.set_title('S11 Smith Chart', color='white')
    _smith_grid(ax)

    s = sim.s11
    ax.plot(s.real, s.imag, '#00ddff', lw=2.2, label='S11 Sim')
    ax.plot(s.real[:1], s.imag[:1], 'o', color='#00ff88', ms=6,
            label=f'Start {sim.freqs[0]/1e6:.0f} MHz')
    ax.plot(s.real[-1:], s.imag[-1:], 's', color='#ff4444', ms=6,
            label=f'Stop {sim.freqs[-1]/1e6:.0f} MHz')
    if g11 is not None:
        ax.plot(g11.real, g11.imag, '#00ddff', lw=1.2, linestyle=':', alpha=0.7, label='S11 Golden')
    if meas and meas.return_gamma is not None:
        g = meas.return_gamma
        ax.plot(g.real, g.imag, '#ffaa00', lw=1.8, linestyle='--', label='S11 Meas')

    lines = []
    for i, m in enumerate(state.markers):
        if not m.enabled or m.freq is None or m.gamma is None:
            continue
        ax.plot(m.gamma.real, m.gamma.imag, 'o', color=m.color, ms=8,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.annotate(f'M{i+1}', (m.gamma.real, m.gamma.imag), xytext=(6, 6),
                    textcoords='offset points', color='white', fontsize=9)
        z = 50.0 * (1 + m.gamma) / (1 - m.gamma) if abs(1 - m.gamma) > 1e-9 else complex('inf')
        lines += [f'M{i+1}: {m.freq/1e6:.3f} MHz',
                  f'   S11 = {m.value:.2f} dB',
                  f'   Γ = {m.gamma.real:+.3f}{m.gamma.imag:+.3f}j',
                  f'   Z = {z.real:.1f}{z.imag:+.1f}j Ω']
    if deltas:
        lines.append('--- Δ ---')
        for i1, i2, df, dv, unit in deltas:
            lines += [f'M{i1}-M{i2}: {df:+.3f} MHz', f'   Δ = {dv:+.3f}{unit}']
    if lines:
        ax.text(0.99, 0.99, 'MARKERS\n' + '\n'.join(lines), transform=ax.transAxes, color='white',
                fontsize=9, fontfamily='monospace', va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117',
                          edgecolor='#555555', alpha=0.95))

    leg = ax.legend(loc='upper left')
    leg.get_frame().set_facecolor('#0d1117')
    for t in leg.get_texts():
        t.set_color('white')

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor='#1a1a1a')
    fig.clf()
    return buf.getvalue()


# ===================== VIEW + CONTROLLER =====================

class CAT(ui.View):
    def __init__(self):
        self.background_color = 'black'
        self.flex = 'WH'
        self._token = 0
        self._drawing = False
        self.state = CATState()
        synthesize(self.state)

        self.status = ui.Label()
        self.status.text_color = '#aaaaaa'
        self.status.font = ('<system>', 11)
        self.status.alignment = ui.ALIGN_CENTER
        self.status.background_color = '#111111'
        self.add_subview(self.status)

        self.img = ui.ImageView()
        self.img.flex = 'WH'
        self.img.content_mode = ui.CONTENT_SCALE_ASPECT_FIT
        self.img.background_color = 'black'
        self.add_subview(self.img)

        self._build_menus()
        self.update_status()
        self.redraw()

    # ---------- menus ----------
    def _build_menus(self):
        menus = {
            'main': [
                ('Edit', self.edit_matrix), ('Undo', self.undo), ('Reset', self.reset),
                ('Random', self.randomize), ('Design', self.edit_parameters),
                ('Scales', self.set_limits), ('Zoom Band', self.zoom_to_band),
                ('Ch1', lambda: self._cycle('ch1_mode', ['s11', 's22', 'smith', 'none'])),
                ('Ch2', lambda: self._cycle('secondary_plot', ['loss', 'phase', 'gd', 'none'])),
                ('Golden', self.toggle_golden_ref),
                ('TZ', lambda: self.show_menu('tz')),
                ('Analysis', lambda: self.show_menu('analysis')),
                ('S2P', lambda: self.show_menu('files')),
                ('Markers', lambda: self.show_menu('markers')),
                ('Help', lambda: show_text('Help / Getting Started', HELP_TEXT)),
            ],
            'tz': [
                ('Design TZs', self.design_tzs), ('TZ Info', self.show_tzs),
                ('Clear TZs', self.clear_tzs),
            ],
            'analysis': [
                ('Diagnostics', self.run_diagnostics), ('Poles / GD', self.show_poles),
                ('Sensitivity', self.run_sensitivity_analysis),
            ],
            'files': [
                ('Save s2p', self.export_s2p), ('Load s2p', self.load_s2p),
                ('Toggle s2p', self.toggle_meas), ('Rel Phase', self.update_phase_flatten),
                ('Clear s2p', self.clear_meas), ('Delete', self.delete_file),
            ],
            'markers': [
                ('Marker 1', lambda: self.set_marker(0)), ('Marker 2', lambda: self.set_marker(1)),
                ('Marker 3', lambda: self.set_marker(2)), ('Clear Markers', self.clear_markers),
            ],
        }
        self.menu_buttons = {}
        for name, defs in menus.items():
            if name != 'main':
                defs = defs + [('Return', lambda: self.show_menu('main'))]
            self.menu_buttons[name] = []
            for title, action in defs:
                b = ui.Button(title=title)
                b.tint_color = '#00ccff'
                b.border_width = 1
                b.border_color = '#444'
                b.corner_radius = 6
                b.action = lambda sender, f=action: f()
                b.hidden = name != 'main'
                self.menu_buttons[name].append(b)
                self.add_subview(b)

    def show_menu(self, name='main'):
        for menu, buttons in self.menu_buttons.items():
            for b in buttons:
                b.hidden = menu != name
        self.layout()

    def layout(self):
        status_h = 22
        self.status.frame = (0, 0, self.width, status_h)
        visible = [b for bs in self.menu_buttons.values() for b in bs if not b.hidden]
        btn_w, gap = 100, 5
        per_row = max(1, int((self.width - 16) / (btn_w + gap)))
        rows = (len(visible) + per_row - 1) // per_row
        btn_h = 8 + rows * 36 + (rows - 1) * 3
        for i, b in enumerate(visible):
            row, col = divmod(i, per_row)
            b.frame = (8 + col * (btn_w + gap), status_h + 4 + row * 36, btn_w, 32)
        self.img.frame = (0, status_h + btn_h, self.width, self.height - status_h - btn_h)

    def update_status(self):
        s = self.state
        modified = s.sim_dirty or (s.M_current is not None and s.M_golden is not None
                                   and not np.allclose(s.M_current, s.M_golden, atol=1e-8))
        self.status.text = (f'n={s.n}  f0={s.f0/1e6:.1f}  BW={s.bw_design/1e6:.1f}  '
                            f'Qu={s.qu:.0f}  {"B" if s.prototype == "butterworth" else "C"}'
                            f'{" *" if modified else ""}'
                            f'{"  TZ:%d" % len(s.tz_freqs) if s.tz_freqs else ""}')

    # ---------- drawing (debounced by token) ----------
    def redraw(self, sender=None, immediate=False):
        self._token += 1
        token = self._token
        ui.cancel_delays()
        if immediate:
            self._perform_redraw()
        else:
            ui.delay(lambda: self._token == token and self._perform_redraw(), 0.35)

    def _perform_redraw(self):
        if self._drawing:
            return
        self._drawing = True
        s = self.state
        try:
            sim = simulate(s)
            meas = prepare_meas(s, sim.freqs)
            for m in s.markers:
                if m.enabled and m.freq is not None:
                    m.update(sim, s)
            golden = (golden_traces(s, sim) if s.show_golden_ref and s.M_golden is not None
                      else (None, None))
            data = render_plot(s, sim, meas, golden, marker_deltas(s.markers))

            self.img.image = ui.Image.from_data(data)
            if meas and not meas.overlap_ok:
                hud('Meas frequency overlap is poor')
            if sim.cond_max is not None and sim.cond_max > COND_WARN:
                hud(f'High cond ≈ {sim.cond_max:.1e}')
            self.update_status()
        except Exception as e:
            alert('Plot Render Error', str(e))
        finally:
            self._drawing = False
            gc.collect()

    def will_close(self):
        ui.cancel_delays()
        plt.close('all')

    def _changed(self):
        """Matrix edited: redraw now and refresh the status bar."""
        self.state.mark_dirty()
        self.redraw(immediate=True)
        self.update_status()

    # ---------- matrix editing ----------
    def _reset_to_golden(self):
        s = self.state
        s.push_undo()
        s.M_current = s.M_golden.copy()
        s.mark_dirty()

    def undo(self):
        if self.state.undo():
            self.redraw()
        else:
            hud('Nothing to undo')

    def reset(self):
        if not confirm('Confirm', 'Reset all changes?'):
            return
        s = self.state
        s.phase_flatten = False
        s.phase_delay = s.phase_constant = 0.0
        self._reset_to_golden()
        self._wipe_markers()
        self.redraw()

    def randomize(self):
        if not confirm('Confirm', 'Randomize matrix?'):
            return
        self._reset_to_golden()
        M, n = self.state.M_current, self.state.n
        for i in range(1, n):
            M[i, i + 1] += random.uniform(-0.08, 0.08)
            M[i + 1, i] = M[i, i + 1]
        for i, j in cross_pairs(n):
            if abs(M[i, j]) > 1e-9:
                M[i, j] += random.uniform(-0.03, 0.03)
                M[j, i] = M[i, j]
        for i in range(1, n + 1):
            M[i, i] += random.uniform(-0.05, 0.05)
        self.state.mark_dirty()
        self.redraw()

    def edit_matrix(self):
        s = self.state
        M = s.M_current
        labels = build_labels(s.n)
        fields = []
        for lbl in labels:
            i, j = label_to_ij(s.n, lbl)
            delta = M[i, j] - s.M_golden[i, j]
            fields.append({'title': f'{lbl} Δ={delta:+.4f}', 'type': 'number',
                           'value': repr(float(M[i, j]))})
        res = form('Edit Matrix (Δ vs Golden)', fields)
        if not res:
            return
        changed = False
        for lbl in labels:
            i, j = label_to_ij(s.n, lbl)
            key = next((k for k in res if k.split()[0] == lbl), None)
            if key is None:
                continue
            try:
                new = float(res[key])
            except Exception:
                continue
            if abs(new - M[i, j]) > 1e-6:
                if not changed:
                    s.push_undo()
                    changed = True
                M[i, j] = M[j, i] = new
        if changed:
            errors = validate_matrix(s, s.M_current, 'M_current')
            if errors:
                alert('Matrix Validation', '\n'.join(errors))
            self._changed()

    # ---------- design / plot settings ----------
    def edit_parameters(self):
        s = self.state
        fields = [
            {'title': 'Number of resonators (n)', 'type': 'number', 'value': str(s.n)},
            {'title': 'Center frequency (MHz)', 'type': 'number', 'value': f'{s.f0/1e6:.1f}'},
            {'title': 'Design bandwidth (MHz)', 'type': 'number', 'value': f'{s.bw_design/1e6:.1f}'},
            {'title': 'Unloaded Q (Qu)', 'type': 'number', 'value': f'{s.qu:.0f}'},
            {'title': 'Number of points', 'type': 'number', 'value': str(s.num_points)},
            {'title': 'Passband ripple (dB)', 'type': 'number', 'value': f'{s.ripple_dB:.3f}'},
            {'title': 'Prototype (cheby/butter)', 'type': 'text', 'value': s.prototype},
            {'title': 'De-embed delay (ps)', 'type': 'number', 'value': f'{s.deembed_delay*1e12:.1f}'},
            {'title': TZ_TITLE, 'type': 'text', 'value': self._tz_text()},
        ]
        res = form('Edit Filter Parameters', fields)
        if not res:
            return
        try:
            new_n = max(2, int(float(res['Number of resonators (n)'])))
            f0 = float(res['Center frequency (MHz)']) * 1e6
            bw = float(res['Design bandwidth (MHz)']) * 1e6
            qu = float(res['Unloaded Q (Qu)'])
            ripple = float(res['Passband ripple (dB)'])
            points = max(100, min(3001, int(float(res['Number of points']))))
            proto = ('butterworth' if res.get('Prototype (cheby/butter)', '').strip().lower()
                     .startswith('butter') else 'chebyshev')
            deembed_s = float(res.get('De-embed delay (ps)', '0') or 0) * 1e-12
            tz = parse_tz_text(res.get(TZ_TITLE, ''))
        except Exception as e:
            alert('Input Error', str(e))
            return
        self._apply_design(new_n, f0, bw, qu, points, ripple, proto, deembed_s, tz)

    def _tz_text(self):
        return ', '.join(f'{f/1e6:.3f}' for f in self.state.tz_freqs)

    def _apply_design(self, n, f0, bw, qu, points, ripple, proto, deembed_s, tz_freqs):
        """Apply design parameters and resynthesize; restores the old design on failure."""
        s = self.state
        err = check_tzs(n, tz_freqs, f0, bw / f0)
        if err:
            alert('Transmission Zeros', err)
            return
        if tz_freqs:
            proto = 'chebyshev'                     # zeros use the Chebyshev-type function
        keys = ['n', 'f0', 'bw_design', 'fbw_design', 'qu', 'num_points', 'ripple_dB', 'prototype',
                'deembed_delay', 'tz_freqs', 'g', 'Qe_source', 'Qe_load', 'M_golden', 'M_current']
        snapshot = {k: getattr(s, k) for k in keys}
        old_n = s.n
        s.n, s.f0, s.bw_design, s.fbw_design = n, f0, bw, bw / f0
        s.qu = max(MIN_QU, min(MAX_QU, qu))
        s.num_points, s.ripple_dB, s.prototype = points, ripple, proto
        s.deembed_delay = deembed_s
        s.tz_freqs = sorted(tz_freqs)
        try:
            synthesize(s)
        except Exception as e:
            for k, v in snapshot.items():
                setattr(s, k, v)
            alert('Synthesis Error', str(e))
            return
        s.undo_stack.clear()
        lo, hi = s.f0 - 1.5 * s.bw_design, s.f0 + 1.5 * s.bw_design
        if s.tz_freqs:
            lo = min(lo, s.tz_freqs[0] - 0.3 * s.bw_design)
            hi = max(hi, s.tz_freqs[-1] + 0.3 * s.bw_design)
        if (not (s.start_freq < s.f0 < s.stop_freq)
                or (s.tz_freqs and not (s.start_freq < s.tz_freqs[0] and s.tz_freqs[-1] < s.stop_freq))):
            s.start_freq, s.stop_freq = max(1e6, lo), hi
        s.mark_dirty()
        if old_n != n:
            alert('Order Changed', f'Filter order changed from {old_n} to {n}.\nMatrix reset.')
        self.redraw()

    def design_tzs(self):
        s = self.state
        fields = [
            {'title': TZ_TITLE, 'type': 'text', 'value': self._tz_text()},
            {'title': 'Passband ripple (dB)', 'type': 'number', 'value': f'{s.ripple_dB:.3f}'},
        ]
        res = form(f'Design TZs (n={s.n}: up to {max(s.n - 2, 0)} zeros)', fields)
        if not res:
            return
        try:
            tz = parse_tz_text(res[TZ_TITLE])
            ripple = float(res['Passband ripple (dB)'])
        except Exception as e:
            alert('Input Error', str(e))
            return
        self._apply_design(s.n, s.f0, s.bw_design, s.qu, s.num_points, ripple, s.prototype,
                           s.deembed_delay, tz)

    def clear_tzs(self):
        s = self.state
        if not s.tz_freqs:
            hud('No transmission zeros set')
        elif confirm('Clear TZs', 'Return to the all-pole prototype? Matrix edits are lost.'):
            self._apply_design(s.n, s.f0, s.bw_design, s.qu, s.num_points, s.ripple_dB,
                               s.prototype, s.deembed_delay, [])

    def show_tzs(self):
        s = self.state
        try:
            w = np.atleast_1d(matrix_tz_omegas(s.n, s.M_current))
        except Exception as e:
            alert('TZ Info', str(e))
            return
        design = sorted(s.tz_freqs)
        real = sorted(x.real for x in w if abs(x.imag) < 1e-5)
        cur = [omega_to_freq(x, s.f0, s.fbw_design) for x in real]
        mhz = lambda fs: ',  '.join(f'{f/1e6:.2f}' for f in fs) + ' MHz'
        lines = []
        if design:
            lines.append('Designed:  ' + mhz(design))
        lines.append('Current:  ' + (mhz(cur) if cur else 'none'))
        if cur and len(cur) == len(design):
            lines.append('Shift:  ' + ',  '.join(f'{(c - d)/1e6:+.2f}' for c, d in zip(cur, design)) + ' MHz')
        for x in w:
            if abs(x.imag) >= 1e-5 and x.imag > 0:
                lines.append(f'Complex pair:  Ω = {x.real:+.3f} ± j{x.imag:.3f}')
        show_text('Transmission Zeros', '\n'.join(lines))

    def set_limits(self):
        s = self.state
        fields = []
        for title, attr, scale, fmt, _ in LIMIT_FIELDS:
            v = getattr(s, attr)
            text = '' if v is None else (fmt.format(v / scale) if fmt else str(v))
            fields.append({'title': title, 'type': 'number', 'value': text})
        res = form('Plot Limits', fields)
        if not res:
            return
        try:
            new = {}
            for title, attr, scale, _, optional in LIMIT_FIELDS:
                txt = str(res[title]).strip()
                new[attr] = None if (optional and not txt) else float(txt) * scale
        except ValueError as e:
            alert('Error', str(e))
            return
        for attr, v in new.items():
            setattr(s, attr, v)
        s.mark_dirty()
        self.redraw()

    def zoom_to_band(self):
        s = self.state
        s.start_freq = max(1e6, s.f0 - 1.5 * s.bw_design)
        s.stop_freq = s.f0 + 1.5 * s.bw_design
        s.mark_dirty()
        self.redraw()
        hud('Sweep set to ~3× BW')

    def _cycle(self, attr, modes):
        cur = getattr(self.state, attr)
        setattr(self.state, attr, modes[((modes.index(cur) if cur in modes else 0) + 1) % len(modes)])
        self.redraw()

    def toggle_golden_ref(self):
        self.state.show_golden_ref = not self.state.show_golden_ref
        self.redraw()
        hud('Golden ON' if self.state.show_golden_ref else 'Golden OFF')

    # ---------- markers ----------
    def _wipe_markers(self):
        for m in self.state.markers:
            m.freq, m.enabled, m.value, m.gamma = None, False, None, None

    def set_marker(self, index):
        text = ask('Set Marker', f'Enter frequency for Marker {index+1} (MHz):', '')
        if text:
            try:
                self.state.markers[index].freq = float(text) * 1e6
                self.state.markers[index].enabled = True
            except ValueError:
                alert('Error', 'Invalid frequency.')
        self.redraw()

    def clear_markers(self):
        self._wipe_markers()
        self.redraw()

    # ---------- analysis ----------
    def run_diagnostics(self):
        try:
            show_text('RF Diagnostics', run_all_diagnostics(self.state))
        except Exception as e:
            alert('Diagnostics Error', str(e))

    def show_poles(self):
        poles = compute_poles_approx(self.state)
        if poles is None:
            alert('Poles', 'Could not compute poles.')
            return
        lines = ['Approximate poles (narrowband mapping):\n']
        lines += [f'P{k}:  {p.real/1e6:.3f} MHz   (σ ≈ {p.imag/1e6:.3f} MHz)'
                  for k, p in enumerate(poles, 1)]
        try:
            gdv = group_delay_variation(self.state, simulate(self.state))
            if gdv is not None:
                lines.append(f'\nPassband GD variation ≈ {gdv:.2f} ns')
        except Exception:
            pass
        show_text('Poles & GD Variation', '\n'.join(lines))

    def run_sensitivity_analysis(self):
        hud('Running sensitivity...')
        show_text('Sensitivity Analysis', run_sensitivity(self.state))

    # ---------- files ----------
    def export_s2p(self):
        s = self.state
        name = ask('Save s2p', 'Filename:', f'filter_n{s.n}' + ('_tz' if s.tz_freqs else ''))
        if not name:
            return
        if not name.endswith('.s2p'):
            name += '.s2p'
        try:
            freqs = np.linspace(s.start_freq, s.stop_freq, s.num_points)
            s11, s21, s12, s22 = compute_s(freqs, s.M_current, s)
            write_s2p(os.path.join(s.DOCS, name), freqs, s11, s21, s12, s22)
            alert(f'Exported {name}')
        except Exception as e:
            alert('Export Error', str(e))
        self.redraw()

    def load_s2p(self):
        s = self.state
        files = sorted(f for f in os.listdir(s.DOCS) if f.lower().endswith('.s2p'))
        if not files:
            alert('No Files', 'No .s2p files found in S2P folder.')
            return
        choice = pick('Load s2p', files)
        if not choice:
            return
        s.loaded_s2p_path = os.path.join(s.DOCS, choice)
        s.loaded_freqs, s.loaded_s11, s.loaded_s21, s.loaded_s22 = read_s2p(s.loaded_s2p_path)
        if s.loaded_freqs is None:
            alert('Parse Error', 'Could not read .s2p file.')
            return
        if len(s.loaded_freqs) > 1:
            lo, hi = s.loaded_freqs[0], s.loaded_freqs[-1]
            if hi < s.start_freq or lo > s.stop_freq:
                alert('Frequency Overlap', 'Loaded S2P does not overlap current sweep.')
            elif (hi - lo) < 0.5 * (s.stop_freq - s.start_freq):
                hud('Partial frequency overlap with sweep')
        if s.loaded_s11 is not None:
            s.loaded_s11 = deembed(s.loaded_freqs, s.loaded_s11, s.deembed_delay)
        if s.loaded_s22 is not None:
            s.loaded_s22 = deembed(s.loaded_freqs, s.loaded_s22, s.deembed_delay)
        s.show_meas = True
        self.redraw()

    def toggle_meas(self):
        if self.state.loaded_freqs is not None:
            self.state.show_meas = not self.state.show_meas
            self.redraw()

    def clear_meas(self):
        self.state.clear_meas()
        self.redraw()

    def delete_file(self):
        s = self.state
        files = [f for f in os.listdir(s.DOCS) if f.endswith(('.npz', '.s2p'))]
        if not files:
            alert('No Files', 'No files found.')
            return
        choice = pick('Delete File', files)
        if not choice or not confirm('Confirm Delete', f'Delete {choice}?'):
            return
        try:
            os.remove(os.path.join(s.DOCS, choice))
            hud(f'Deleted {choice}')
            if s.loaded_s2p_path and os.path.basename(s.loaded_s2p_path) == choice:
                s.clear_meas()
        except Exception as e:
            alert('Delete Error', str(e))
        self.redraw()

    def update_phase_flatten(self):
        """Fit a linear phase to the golden S21 across the design band."""
        s = self.state
        try:
            freqs = np.linspace(s.start_freq, s.stop_freq, s.num_points)
            phase_deg = unwrap_phase(compute_s(freqs, s.M_golden, s)[1]) / np.pi * 180
            i1 = np.argmin(np.abs(freqs - (s.f0 - s.bw_design / 2)))
            i2 = np.argmin(np.abs(freqs - (s.f0 + s.bw_design / 2)))
            slope = (phase_deg[i2] - phase_deg[i1]) / (freqs[i2] - freqs[i1])
            s.phase_delay = -slope / 360.0
            s.phase_constant = -(phase_deg[i1] + 360 * freqs[i1] * s.phase_delay)
            s.phase_flatten = True
            s.mark_dirty()
            self.redraw()
        except Exception as e:
            alert('Phase Flatten Error', str(e))


# ===================== MAIN =====================

if __name__ == '__main__':
    CAT().present('fullscreen', orientations=['landscape_left', 'landscape_right'])