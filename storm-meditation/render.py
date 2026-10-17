"""
Procedural storm animation for the meditating-figure painting.

Everything is classic image processing (numpy + OpenCV + ffmpeg), no generative AI:
  * GrabCut separates the figure and her hair from the landscape.
  * The figure stays perfectly still; only hair, sky, mist, dust and light change.
  * Clouds/mist are fBm value noise, re-textured with the painting's own paper grain.
  * Hair moves with a smooth displacement field anchored at the scalp.
  * Clothes pick up wind-blown dust stains that build up over time.
  * A wind soundtrack is synthesised from filtered noise driven by the same gusts.

Usage: python3 render.py [--preview]
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source.png")
OUT = os.path.join(HERE, "storm_meditation.mp4")
CACHE = os.path.join(HERE, ".masks.npz")

PREVIEW = "--preview" in sys.argv
W, H = (540, 960) if PREVIEW else (1080, 1920)
FPS = 30
DUR = 15.0
N = int(FPS * DUR)
SRC_W, SRC_H = 941, 1672
S = W / SRC_W  # source-pixel -> output-pixel scale

rng_master = np.random.default_rng(7)


# ----------------------------------------------------------------------------- helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fbm(h, w, seed, octaves=6, base=3.0, persistence=0.5, aniso=1.0):
    """Value-noise fBm in [0,1]. aniso > 1 stretches features horizontally."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        f = base * (2 ** o)
        gh = int(f) + 2
        gw = int(f * w / h / aniso) + 2
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    out /= total
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo), 0, 1)


def noise1d(n, seed, knots):
    """Smooth 1D noise sampled at n points, values in [-1, 1]."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(-1, 1, knots + 4)
    x = np.linspace(1, knots + 1, n)
    from scipy.interpolate import CubicSpline
    return np.clip(CubicSpline(np.arange(knots + 4), k)(x), -1, 1)


def luma(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def over(dst, rgb, a):
    a = a[..., None] if a.ndim == 2 else a
    return dst * (1 - a) + rgb * a


# ----------------------------------------------------------------------------- masks
def build_masks():
    im = cv2.imread(SRC)
    mask = np.full(im.shape[:2], cv2.GC_BGD, np.uint8)
    mask[270:1470, 215:660] = cv2.GC_PR_BGD
    prob = np.array([[440, 340], [520, 360], [570, 520], [620, 700], [625, 1000], [600, 1250],
                     [540, 1390], [480, 1450], [420, 1440], [400, 1380], [370, 1130], [240, 960],
                     [250, 860], [290, 700], [330, 520], [380, 380]])
    cv2.fillPoly(mask, [prob], cv2.GC_PR_FGD)
    sure = np.array([[450, 380], [500, 420], [560, 650], [590, 900], [570, 1200], [500, 1350],
                     [440, 1350], [420, 1100], [300, 920], [330, 650], [400, 450]])
    cv2.fillPoly(mask, [sure], cv2.GC_FGD)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(im, mask, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
    fig = ((mask == 1) | (mask == 3)).astype(np.uint8)
    # GrabCut grabs a sunlit rock patch beside the standing foot; keep only the foot there.
    foot = np.zeros_like(fig)
    cv2.ellipse(foot, (457, 1408), (34, 52), 0, 0, 360, 1, -1)
    fig[1392:] &= foot[1392:]
    fig = cv2.morphologyEx(fig, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fig)
    fig = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

    # Hair = dark pixels of the figure inside the hair zones (not the rudraksha beads).
    rgb = im[..., ::-1].astype(np.float32) / 255
    dark = (luma(rgb) < 0.42).astype(np.uint8)
    zone = np.zeros_like(fig)
    cv2.fillPoly(zone, [np.array([[370, 330], [520, 330], [530, 380], [470, 360], [420, 365], [370, 395]])], 1)
    cv2.fillPoly(zone, [np.array([[372, 360], [405, 362], [408, 495], [410, 880], [250, 880], [262, 600], [330, 400]])], 1)
    cv2.fillPoly(zone, [np.array([[505, 350], [560, 380], [600, 470], [600, 580], [532, 580], [518, 470], [505, 420]])], 1)
    hair = dark & zone & fig
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    hair &= zone & fig
    np.savez_compressed(CACHE, fig=fig, hair=hair)
    return fig, hair


def load_masks():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["fig"], d["hair"]
    return build_masks()


# ----------------------------------------------------------------------------- setup
print(f"setup {W}x{H}, {N} frames")
src8 = cv2.imread(SRC)
src8 = cv2.resize(src8, (W, H), interpolation=cv2.INTER_LANCZOS4)
fig_n, hair_n = load_masks()
fig = cv2.resize(fig_n.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
hair = cv2.resize(hair_n.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
fig = cv2.GaussianBlur(fig, (0, 0), 0.8 * S)
hair = np.clip(cv2.GaussianBlur(hair, (0, 0), 0.7 * S) * 1.15, 0, 1)

# Plate with the hair painted out, so moving strands reveal plausible background.
hair_hole = cv2.dilate((hair > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8))
plate8 = cv2.inpaint(src8, hair_hole, 6 * S, cv2.INPAINT_TELEA)
src = src8[..., ::-1].astype(np.float32) / 255
plate = plate8[..., ::-1].astype(np.float32) / 255
# Pixels hidden under the hair belong to whichever side (body or sky) is nearer,
# so a strand swaying aside reveals sleeve next to the sleeve and sky next to the sky.
body_known = ((fig > 0.5) & (hair_hole == 0)).astype(np.uint8)
bg_known = ((fig <= 0.5) & (hair_hole == 0)).astype(np.uint8)
d_body = cv2.distanceTransform(1 - body_known, cv2.DIST_L2, 5)
d_bg = cv2.distanceTransform(1 - bg_known, cv2.DIST_L2, 5)
body_side = ((hair_hole > 0) & (d_body < d_bg)).astype(np.float32)
body = np.clip(fig * (1 - hair_hole) + body_side, 0, 1)
body = np.maximum(cv2.GaussianBlur(body, (0, 0), 0.8 * S) * (hair_hole > 0), fig * (1 - hair_hole))

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
ys = yy / S  # in source pixels
xs = xx / S

# Grass and trees in the foreground: olive pixels with vertical blade structure.
hsv = cv2.cvtColor(src8, cv2.COLOR_BGR2HSV)
olive = ((hsv[..., 0] >= 17) & (hsv[..., 0] <= 32) & (hsv[..., 1] > 70)).astype(np.float32)
olive = cv2.GaussianBlur(olive, (0, 0), 3 * S)
_L8 = src8.astype(np.float32).mean(2) / 255
_gx = cv2.GaussianBlur(np.abs(cv2.Sobel(_L8, cv2.CV_32F, 1, 0, ksize=3)), (0, 0), 3 * S)
_gy = cv2.GaussianBlur(np.abs(cv2.Sobel(_L8, cv2.CV_32F, 0, 1, ksize=3)), (0, 0), 3 * S)
veg = np.clip(olive * 1.3, 0, 1) * np.clip((_gx / (_gy + 1e-3) - 0.9) * 1.5, 0, 1)
veg_y0 = int(1120 * S)
veg = veg * smoothstep(1120, 1180, np.mgrid[0:H, 0:W][0] / S)
veg = veg * (1 - cv2.dilate((fig > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)))
veg = cv2.GaussianBlur(veg.astype(np.float32), (0, 0), 2 * S)[veg_y0:]
vy, vx = np.mgrid[veg_y0:H, 0:W].astype(np.float32)

# Paper/brush grain of the painting itself, reused on everything we paint.
L = luma(src)
grain = np.clip(L / (cv2.GaussianBlur(L, (0, 0), 6 * S) + 1e-3), 0.75, 1.25).astype(np.float32)
grain3 = grain[..., None]

# Sky weight: full in the sky, thinning over the far peaks, gone by the valley mist.
sky_w = 1 - smoothstep(560, 980, ys)
sky_w = sky_w * (0.55 + 0.45 * (1 - smoothstep(560, 760, ys)))

# Halo glow around her head stays softly lit through the storm.
hc = (447 * S, 423 * S)
hr = np.hypot(xx - hc[0], yy - hc[1]) / S
halo_keep = np.exp(-((hr - 132) / 26) ** 2) * 0.9 + np.exp(-(hr / 150) ** 2) * 0.55
halo_keep = np.clip(halo_keep, 0, 1)

# Sun disc (fades out as the clouds swallow it).
sun_r = np.hypot(xs - 856, ys - 622)
sun_glow = np.exp(-(sun_r / 70) ** 2)

# Dirt on clothes: windward (right) side and the lower hem collect the most dust.
skin = np.zeros((SRC_H, SRC_W), np.uint8)
cv2.ellipse(skin, (450, 420), (58, 80), 0, 0, 360, 1, -1)            # face + neck
cv2.ellipse(skin, (458, 1405), (40, 55), 0, 0, 360, 1, -1)           # standing foot
cv2.ellipse(skin, (448, 965), (30, 70), -15, 0, 360, 1, -1)          # raised foot
cv2.fillPoly(skin, [np.array([[420, 520], [500, 520], [505, 700], [415, 700]])], 1)  # hands
skin = cv2.resize(skin.astype(np.float32), (W, H))
skin = cv2.GaussianBlur(skin, (0, 0), 3 * S)
cloth = np.clip(body * (1 - skin), 0, 1)
dirt_tex = fbm(H, W, 11, octaves=6, base=6, aniso=3.0)
dirt_fine = fbm(H, W, 12, octaves=4, base=40, aniso=2.0)
dirt_where = 1.0 * smoothstep(1000, 1430, ys) + 0.45 * smoothstep(520, 620, xs) * smoothstep(700, 1100, ys)
dirt_score = dirt_tex * 0.7 + dirt_fine * 0.3
# Speckles of grit that land one by one.
n_spk = int(1400 * (W / 1080))
spk_y = rng_master.uniform(600, 1450, n_spk)
spk_x = rng_master.uniform(240, 630, n_spk)
keep = (rng_master.random(n_spk) < (0.08 + 0.92 * smoothstep(1000, 1440, spk_y)))
spk_x, spk_y = spk_x[keep], spk_y[keep]
spk_t = rng_master.uniform(3.0, 14.0, len(spk_x))
spk_r = rng_master.uniform(0.6, 1.6, len(spk_x))

# Cloud fields (half/quarter res, wide enough to scroll for the whole clip).
CQ = 4
ch, cw = H // CQ, W // CQ
cloud_wide = 3 * cw
cloud_a = fbm(ch, cloud_wide, 21, octaves=6, base=3, aniso=2.2)
cloud_b = fbm(ch, cloud_wide, 22, octaves=6, base=4, aniso=1.6)
warp_x = (fbm(ch, cloud_wide, 23, octaves=3, base=2) - 0.5) * 18
warp_y = (fbm(ch, cloud_wide, 24, octaves=3, base=2) - 0.5) * 10
mist_tex = fbm(ch, cloud_wide, 25, octaves=5, base=3, aniso=5.0)
cyq, cxq = np.mgrid[0:ch, 0:cw].astype(np.float32)
_d = cloud_a * 0.6 + cloud_b * 0.4
dens_lo, dens_hi = np.percentile(_d, 2), np.percentile(_d, 98)
strokes = fbm(H, W * 3, 26, octaves=3, base=30, aniso=7.0)

# Timeline.
t_all = np.arange(N) / FPS
storm_curve = smoothstep(1.5, 10.0, t_all)
gust_n = noise1d(N, 31, 18)
wind_curve = (0.25 + 0.75 * smoothstep(0.5, 9.0, t_all)) * (0.78 + 0.22 * gust_n)
# Integrated "wind distance", for things that should speed up smoothly with gusts.
wind_dist = np.cumsum(wind_curve) / FPS

# Hair displacement setup (only evaluated inside the hair bounding box).
ys0, ys1 = int(310 * S), int(900 * S)
xs0, xs1 = int(225 * S), int(640 * S)
hy, hx = np.mgrid[ys0:ys1, xs0:xs1].astype(np.float32)
head = (446 * S, 402 * S)
hr_head = np.hypot(hx - head[0], hy - head[1]) / S
hair_w = smoothstep(85, 470, hr_head) ** 1.15
hair_flutter = [fbm(ys1 - ys0, (xs1 - xs0) * 3, 40 + i, octaves=3, base=3) - 0.5 for i in range(2)]

# Particles.
rng = np.random.default_rng(99)


class Dust:
    def __init__(self, n, near):
        self.n = n
        self.x = rng.uniform(0, W * 1.3, n)
        self.y = rng.uniform(0, H, n)
        self.z = rng.uniform(0.4, 1.0, n) if near else rng.uniform(0.15, 0.45, n)
        self.seed = rng.uniform(0, 100, n)
        self.dark = rng.random(n) < 0.7
        self.on = rng.random(n)

    def step(self, t, wind, dt):
        vx = -(260 + 1300 * self.z) * wind * S
        vy = (np.sin(t * 2.3 + self.seed) * 60 + np.sin(t * 5.1 + self.seed * 2) * 30 - 25) * self.z * S
        px, py = self.x.copy(), self.y.copy()
        self.x += vx * dt
        self.y += vy * dt
        out = (self.x < -40) | (self.y < -40) | (self.y > H + 40)
        k = out.sum()
        if k:
            self.x[out] = W + rng.uniform(10, 300, k)
            self.y[out] = rng.uniform(0, H, k)
            px[out], py[out] = self.x[out], self.y[out]
        return px, py


dust_far = Dust(int(220), near=False)
dust_near = Dust(int(120), near=True)


class Petals:
    def __init__(self, n):
        self.n = n
        self.x = np.full(n, -999.0)
        self.y = np.zeros(n)
        self.ang = rng.uniform(0, 360, n)
        self.spin = rng.uniform(-500, 500, n)
        self.size = rng.uniform(2.2, 4.0, n) * S
        self.seed = rng.uniform(0, 100, n)
        self.next = rng.uniform(1.0, 6.0, n)

    def step(self, t, wind, dt):
        for i in range(self.n):
            if self.x[i] < -50 and t >= self.next[i]:
                if rng.random() < 0.55:  # lifted from the meadow
                    self.x[i] = rng.uniform(0.15, 1.0) * W
                    self.y[i] = rng.uniform(1300, 1650) * S
                else:  # blown in from off-screen right
                    self.x[i] = W + 20
                    self.y[i] = rng.uniform(700, 1500) * S
                self.next[i] = t + rng.uniform(0.5, 4.0) / max(wind, 0.3)
        live = self.x > -50
        self.x[live] -= (380 + 900 * wind) * S * dt * (0.7 + 0.3 * np.sin(self.seed[live] + t))
        self.y[live] += (np.sin(t * 3 + self.seed[live]) * 140 - 120 * wind) * S * dt
        self.ang[live] += self.spin[live] * dt
        self.x[self.y < -40] = -999


petals = Petals(28)


class Streaks:
    """Thin, faint gust lines that sweep right-to-left like painted wind."""

    def __init__(self, n):
        self.n = n
        self.x = np.full(n, -1e4)
        self.y = np.zeros(n)
        self.len = np.zeros(n)
        self.amp = np.zeros(n)
        self.ph = np.zeros(n)
        self.spd = np.zeros(n)

    def spawn(self, i):
        self.len[i] = rng.uniform(180, 420) * S
        self.x[i] = W + rng.uniform(0, 400) * S
        self.y[i] = rng.uniform(150, 1550) * S
        self.amp[i] = rng.uniform(6, 22) * S
        self.ph[i] = rng.uniform(0, 6.28)
        self.spd[i] = rng.uniform(900, 1500) * S

    def step(self, t, wind, dt):
        active = int(2 + 8 * wind)
        for i in range(self.n):
            if self.x[i] + self.len[i] < -10:
                if i < active:
                    self.spawn(i)
            else:
                self.x[i] -= self.spd[i] * wind * dt

    def draw(self, layer, t, strength):
        for i in range(self.n):
            if self.x[i] + self.len[i] < 0 or self.x[i] > W:
                continue
            u = np.linspace(0, 1, 40)
            px = self.x[i] + u * self.len[i]
            py = self.y[i] + np.sin(u * 5.0 + self.ph[i] + t * 3.0) * self.amp[i]
            fade = np.sin(np.pi * u) ** 1.5
            for j in range(len(u) - 1):
                a = fade[j] * strength
                if a < 0.01:
                    continue
                cv2.line(layer, (int(px[j] * 4), int(py[j] * 4)), (int(px[j + 1] * 4), int(py[j + 1] * 4)),
                         float(a), max(1, int(S)), cv2.LINE_AA, shift=2)


streaks = Streaks(12)


# ----------------------------------------------------------------------------- per-frame
def grade(img, k, darken, desat, tint):
    g = luma(img)[..., None]
    out = img + (g - img) * (desat * k)
    out = out * (1 - (1 - darken) * k)
    return out * (1 + (np.array(tint, np.float32) - 1) * k)


def clouds(t, storm):
    """Returns (rgb, alpha) of the cloud layer at full res."""
    off_a = 18 * t / CQ * S
    off_b = 34 * t / CQ * S
    evo = t * 0.25
    mx = (cxq + off_a + warp_x[:, :cw] * np.sin(evo)).astype(np.float32)
    my = (cyq + warp_y[:, :cw] * np.cos(evo * 0.8)).astype(np.float32)
    da = cv2.remap(cloud_a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mxb = (cxq + off_b + warp_x[:, cw:2 * cw] * np.cos(evo)).astype(np.float32)
    db = cv2.remap(cloud_b, mxb, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    d = np.clip((da * 0.6 + db * 0.4 - dens_lo) / (dens_hi - dens_lo), 0, 1)
    # Light from the upper right: lit rims where density falls off toward the light.
    sh = cv2.warpAffine(d, np.float32([[1, 0, -3], [0, 1, 4]]), (cw, ch), borderMode=cv2.BORDER_REFLECT)
    lit = cv2.GaussianBlur(np.clip((d - sh) * 7 + 0.5, 0, 1), (0, 0), 1.0)
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    lit = np.clip(cv2.resize(lit, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
    # Storm front arrives from the right.
    local = smoothstep(0.0, 1.0, storm * 1.7 - (1 - xx / W) * 0.7)
    th = 0.78 - 0.42 * local
    a = smoothstep(th, th + 0.10, d)
    core = smoothstep(th + 0.05, th + 0.45, d)[..., None]
    warm_mid = np.array([0.86, 0.74, 0.58], np.float32)
    dark_c = np.array([0.26, 0.225, 0.20], np.float32)
    mid_c = np.array([0.46, 0.41, 0.36], np.float32)
    lite_s = np.array([0.68, 0.60, 0.50], np.float32)
    lite_w = np.array([0.96, 0.84, 0.64], np.float32)
    lite = lite_w + (lite_s - lite_w) * storm
    mid = warm_mid + (mid_c - warm_mid) * storm
    base = mid + (dark_c - mid) * (core * (0.3 + 0.7 * storm))
    rgb = base + (lite - base) * (lit[..., None] ** 1.3 * (1 - 0.5 * core))
    # Overcast sheet behind the cumulus once the storm has settled in.
    sheet_a = smoothstep(0.3, 1.0, local) * 0.8
    sheet_c = np.array([0.35, 0.31, 0.275], np.float32)
    A = a * (0.5 + 0.5 * local)
    out_a = A + sheet_a * (1 - A)
    rgb = (rgb * A[..., None] + sheet_c * (sheet_a * (1 - A))[..., None]) / np.maximum(out_a, 1e-4)[..., None]
    # Brushed look: fine horizontal strokes that travel with the clouds, plus the paper grain.
    so = int(off_b * CQ) % (strokes.shape[1] - W)
    rgb = rgb * (0.9 + 0.2 * strokes[:, so:so + W])[..., None] * grain3
    return rgb, np.clip(out_a * sky_w, 0, 1)


def mist(t, storm, wind_d):
    off = (60 * wind_d) / CQ * S
    m = mist_tex[:, int(off) % (cloud_wide - cw):int(off) % (cloud_wide - cw) + cw]
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)
    band = smoothstep(780, 900, ys) * (1 - smoothstep(1120, 1260, ys))
    band2 = smoothstep(600, 680, ys) * (1 - smoothstep(760, 840, ys)) * 0.6
    a = smoothstep(0.35, 0.85, m) * (band + band2) * (0.15 + 0.45 * storm)
    col = np.array([0.62, 0.57, 0.50], np.float32) * (1 - 0.35 * storm)
    return col * grain3, a


def apply_dirt(img, t):
    amt = smoothstep(2.5, 14.0, t)
    th = 1.05 - 0.6 * amt * dirt_where
    stain = smoothstep(th, th + 0.25, dirt_score) * 0.5 * amt
    film = 0.24 * amt * smoothstep(900, 1440, ys) * (0.6 + 0.4 * dirt_fine)  # dusty haze on the hem
    a = np.clip((stain + film) * cloth, 0, 0.6)[..., None]
    dirt_col = np.array([0.54, 0.42, 0.30], np.float32)
    return img * (1 - a + a * dirt_col)


def draw_speckles(img, t):
    vis = spk_t <= t
    if not vis.any():
        return img
    layer = np.zeros((H, W), np.float32)
    for x, y, r, t0 in zip(spk_x[vis], spk_y[vis], spk_r[vis], spk_t[vis]):
        cv2.circle(layer, (int(x * S * 4), int(y * S * 4)), max(1, int(r * S * 4)), float(min(1, (t - t0) * 3)),
                   -1, cv2.LINE_AA, shift=2)
    a = (layer * cloth * 0.5)[..., None]
    return img * (1 - a) + np.array([0.36, 0.27, 0.19], np.float32) * a


def draw_dust(layer_a, layer_c, d, px, py, wind, alpha_scale):
    for i in range(d.n):
        if d.on[i] > 0.35 + 0.65 * wind:
            continue
        x1, y1 = d.x[i], d.y[i]
        x0, y0 = x1 + (px[i] - x1) * 0.35, y1 + (py[i] - y1) * 0.35
        if x1 < -5 or x0 > W + 5:
            continue
        r = max(1, int((0.6 + 1.6 * d.z[i]) * S))
        a = float(alpha_scale * (0.35 + 0.5 * d.z[i]))
        p0 = (int(x0 * 4), int(y0 * 4))
        p1 = (int(x1 * 4), int(y1 * 4))
        cv2.line(layer_a, p0, p1, a, r, cv2.LINE_AA, shift=2)
        cv2.line(layer_c, p0, p1, 1.0 if d.dark[i] else 0.0, r, cv2.LINE_AA, shift=2)


def composite_dust(img, la, lc, storm):
    la = cv2.GaussianBlur(la, (0, 0), 0.6 * S)
    dark = np.array([0.26, 0.20, 0.15], np.float32)
    light = np.array([0.80, 0.72, 0.60], np.float32) * (1 - 0.3 * storm)
    col = light + (dark - light) * np.clip(lc, 0, 1)[..., None]
    return over(img, col, np.clip(la, 0, 1))


def sway(img, t, wind):
    """Grass and pines bend away from the wind and ripple with the gusts."""
    ripple = np.sin(t * 5.0 + vx * 0.035 / S + vy * 0.01 / S) * 0.5 + np.sin(t * 8.3 + vx * 0.07 / S) * 0.25
    dx = veg * (1.0 + 2.6 * wind) * S * (0.6 + ripple)
    out = img.copy()
    out[veg_y0:] = cv2.remap(img, (vx + dx).astype(np.float32), vy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def frame(i, dt):
    t = t_all[i]
    storm = storm_curve[i]
    wind = wind_curve[i]

    # --- background: graded plate, sun, mist, clouds
    bg = grade(sway(plate, t, wind), storm, darken=0.46, desat=0.5, tint=(0.97, 0.98, 1.02))
    bg = bg - (smoothstep(1100, 0, ys) * 0.12 * storm)[..., None]  # heavier sky at the top
    bg = bg + (sun_glow * 0.25 * (1 - storm))[..., None] * np.array([1.0, 0.85, 0.5], np.float32)
    mc, ma = mist(t, storm, wind_dist[i])
    bg = over(bg, mc, ma)
    cc, ca = clouds(t, storm)
    bg = over(bg, cc, ca)
    # The halo keeps glowing through the storm, in front of the clouds.
    halo_lit = grade(plate, storm, darken=0.8, desat=0.15, tint=(1, 1, 1))
    bg = bg + (halo_lit - bg) * (halo_keep * storm * 0.6)[..., None]
    # cloud shadows drifting over the land
    shadow = cv2.resize(cloud_b[:, int(50 * t / CQ * S):int(50 * t / CQ * S) + cw], (W, H))
    bg = bg * (1 - (smoothstep(0.45, 0.8, shadow) * smoothstep(700, 1000, ys) * 0.18 * storm)[..., None])

    # --- far dust (behind her)
    la = np.zeros((H, W), np.float32)
    lc = np.zeros((H, W), np.float32)
    px, py = dust_far.step(t, wind, dt)
    draw_dust(la, lc, dust_far, px, py, wind, 0.45 + 0.4 * storm)
    bg = composite_dust(bg, la, lc, storm)

    # --- figure (static) with dust building on the cloth
    figc = grade(plate, storm, darken=0.70, desat=0.28, tint=(0.99, 0.99, 1.01))
    figc = figc + (plate - figc) * (halo_keep * storm * 0.25)[..., None]
    figc = apply_dirt(figc, t)
    figc = draw_speckles(figc, t)
    img = over(bg, figc, body)

    # --- hair, moved by the wind
    hsrc = grade(src[ys0:ys1, xs0:xs1], storm, darken=0.70, desat=0.28, tint=(0.99, 0.99, 1.01))
    A = (1.5 + 7.0 * wind) * S
    phase = t * (2.2 + 1.2 * wind)
    wave = np.sin(phase - hr_head * 0.030 + hx * 0.004 / S) * 0.6 + np.sin(phase * 1.7 - hr_head * 0.05 + 1.3) * 0.4
    off = int(t * 40 * S * (0.5 + wind)) % ((xs1 - xs0) * 2)
    fl_x = hair_flutter[0][:, off:off + (xs1 - xs0)]
    fl_y = hair_flutter[1][:, off:off + (xs1 - xs0)]
    # wind blows right -> left: strands lean left and ripple
    dx = A * hair_w * (0.35 + 0.5 * wave + 1.4 * fl_x)
    dy = A * hair_w * (0.25 * wave + 0.8 * fl_y - 0.15)
    mx = (hx + dx - xs0).astype(np.float32)
    my = (hy + dy - ys0).astype(np.float32)
    h_rgb = cv2.remap(hsrc, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    h_a = cv2.remap(hair[ys0:ys1, xs0:xs1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img[ys0:ys1, xs0:xs1] = over(img[ys0:ys1, xs0:xs1], h_rgb, h_a)

    # --- near dust, petals, wind streaks (in front)
    la[:] = 0
    lc[:] = 0
    px, py = dust_near.step(t, wind, dt)
    draw_dust(la, lc, dust_near, px, py, wind, 0.35 + 0.45 * storm)
    img = composite_dust(img, la, lc, storm)

    petals.step(t, wind, dt)
    pl = np.zeros((H, W), np.float32)
    for k in range(petals.n):
        if petals.x[k] < -40 or petals.x[k] > W + 40:
            continue
        c = (int(petals.x[k] * 4), int(petals.y[k] * 4))
        ax = (int(petals.size[k] * 4), int(petals.size[k] * 0.5 * 4 * (0.4 + 0.6 * abs(np.cos(np.radians(petals.ang[k] * 0.7))))))
        cv2.ellipse(pl, c, ax, petals.ang[k], 0, 360, 0.9, -1, cv2.LINE_AA, shift=2)
    img = over(img, np.array([0.93, 0.90, 0.82], np.float32) * (1 - 0.25 * storm), pl)

    streaks.step(t, wind, dt)
    sl = np.zeros((H, W), np.float32)
    streaks.draw(sl, t, 0.10 + 0.14 * wind)
    sl = cv2.GaussianBlur(sl, (0, 0), 0.8 * S)
    img = over(img, np.array([0.88, 0.84, 0.76], np.float32) * (1 - 0.3 * storm), sl)

    # --- gentle storm vignette
    v = ((xx / W - 0.5) ** 2 + (yy / H - 0.45) ** 2 * 0.6)
    img = img * (1 - (v * 1.1 * storm)[..., None])
    return np.clip(img, 0, 1)


# ----------------------------------------------------------------------------- audio
def wind_audio(path, sr=44100):
    n = int(DUR * sr)
    t = np.arange(n) / sr
    env = np.interp(t, t_all, wind_curve)
    rng_a = np.random.default_rng(5)
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        white = rng_a.standard_normal(n)
        # rumble body
        b, a = signal.butter(2, [60 / (sr / 2), 500 / (sr / 2)], "band")
        body_ = signal.lfilter(b, a, white)
        # airy hiss whose brightness follows the gusts (block-wise filtering)
        hiss = np.zeros(n)
        blk = 2048
        zi = None
        for s in range(0, n, blk):
            e = env[min(s, n - 1)]
            lo, hi = 300 + 500 * e, 900 + 2600 * e
            bb, aa = signal.butter(2, [lo / (sr / 2), hi / (sr / 2)], "band")
            if zi is None:
                zi = signal.lfilter_zi(bb, aa) * 0
            hiss[s:s + blk], zi = signal.lfilter(bb, aa, white[s:s + blk], zi=zi)
        # whistle: narrow resonance gliding with the gusts
        whistle = np.zeros(n)
        f0 = 520 + 260 * np.interp(t, t_all, gust_n) + ch * 35
        for s in range(0, n, blk):
            f = f0[s]
            bb, aa = signal.iirpeak(f / (sr / 2), 30)
            whistle[s:s + blk] = signal.lfilter(bb, aa, white[s:s + blk])
        mix = body_ * 0.9 + hiss * (0.35 + 0.9 * env) + whistle * 0.5 * smoothstep(0.5, 1.0, env)
        mix *= (0.15 + 0.85 * env)
        out[:, ch] = mix
    out /= np.abs(out).max() + 1e-6
    fade = np.minimum(1, np.minimum(t / 1.0, (DUR - t) / 1.0))
    out *= (fade * 0.8)[:, None]
    wavfile.write(path, sr, (out * 32767).astype(np.int16))


def main():
    wav = os.path.join(HERE, ".wind.wav")
    wind_audio(wav)
    out = OUT.replace(".mp4", "_preview.mp4") if PREVIEW else OUT
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dt = 1 / FPS
    for i in range(N):
        f = frame(i, dt)
        p.stdin.write((f * 255 + 0.5).astype(np.uint8).tobytes())
        if i % 30 == 0:
            print(f"frame {i}/{N}", flush=True)
            if i % 90 == 0:
                cv2.imwrite(os.path.join(HERE, f".peek_{i:03d}.jpg"), (f[..., ::-1] * 255).astype(np.uint8))
    p.stdin.close()
    p.wait()
    print("wrote", out)


def stills(times):
    dt = 1 / FPS
    want = sorted(int(float(x) * FPS) for x in times.split(","))
    for i in range(max(want) + 1):
        f = frame(i, dt)
        if i in want:
            cv2.imwrite(os.path.join(HERE, f".still_{i:03d}.jpg"), (f[..., ::-1] * 255).astype(np.uint8))


if __name__ == "__main__":
    if "--still" in sys.argv:
        stills(sys.argv[sys.argv.index("--still") + 1])
    else:
        main()
