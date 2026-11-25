#!/usr/bin/env python3
"""
GelSight-Wedge-style 3D reconstruction for vision-based tactile sensors (Meta DIGIT, VBTS, ...).
Wang et al., "GelSight Wedge", ICRA 2021.

    diff RGB + pixel pos (X,Y)  --MLP 5-32-32-32-2-->  (Gx, Gy)  --fast Poisson (DST)-->  depth [mm]

Workflow (see README.md)
    capture  ref (averaged) + calibration-ball presses + end ref            [camera]
    measure  spatial resolution mm/px from clicks on a known distance       [GUI]
    check    background drift / light-leak report                           [no GUI]
    label    manual contact-circle labelling, ball radius stored per image  [GUI]
    review   contact sheet of labels for a second look                      [no GUI]
    train    MLP (RGBXY -> GxGy), Table-I-style gradient errors
    eval     depth vs analytic sphere on the test images
    recon    one image -> depth .npy/.png
    live     real-time depth                                                [camera]
    export   ONNX (MLP as 1x1 convs -> NPU friendly)
    synth    synthetic dataset to dry-run everything without hardware

Camera source (capture / measure --live / live):  --cam N (OpenCV/UVC)  or  --digit SERIAL (digit-interface)

Conventions
    x = column (right), y = row (down); z = indentation depth, positive into the gel.
    Gx = dz/dx, Gy = dz/dy (mm/mm).  depth_mm = poisson(Gx, Gy) * mm_per_px.

sensor.json
    {"width": 320, "height": 240,   # working resolution (after optional unwarp/resize)
     "mm_per_px": null,             # fill with `measure`; must refer to the working resolution
     "corners": null}               # optional [[x,y]*4] TL,TR,BR,BL in raw frame -> unwarp

labels.json (written by `label`), one entry per image:
    {"ball_000.png": {"cx": .., "cy": .., "r": ..,   # contact circle, px (working resolution)
                      "ball_radius_mm": 2.0,           # ball used for THIS press (may differ per image)
                      "depth_mm": null}}               # optional: known press depth (Z-stage) overrides r
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
try:
    from scipy.fft import dstn, idstn
except ImportError:  # scipy < 1.4 (e.g. Jetson apt scipy 1.3.3): fftpack idstn is unnormalized,
    # so use the orthonormal pair, which gives idstn(dstn(x)) == x like scipy.fft's default.
    from scipy import fftpack as _fftpack

    def dstn(x, type=2):
        return _fftpack.dstn(x, type=type, norm='ortho')

    def idstn(x, type=2):
        return _fftpack.idstn(x, type=type, norm='ortho')
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------------- config / IO
def load_sensor(path, need_scale=True):
    with open(path) as f:
        s = json.load(f)
    s.setdefault("corners", None)
    if need_scale and not s.get("mm_per_px"):
        raise SystemExit(f"{path}: mm_per_px is not set. Run `measure` first.")
    return s


def prep(img, s):
    """Raw BGR frame -> working-resolution float32 BGR (unwarp if corners given)."""
    W, H = s.get("width"), s.get("height")
    if not W or not H:  # not set yet -> use frames as they come
        return img.astype(np.float32)
    if s.get("corners"):
        src = np.float32(s["corners"])
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H))
    elif img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def read(path, s):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return prep(img, s)


def ref_image(path, s):
    return cv2.GaussianBlur(read(path, s), (3, 3), 0)


def open_source(a):
    """Returns (read_fn -> BGR uint8 frame or None, close_fn)."""
    if getattr(a, "digit", None):
        from digit_interface import Digit
        d = Digit(a.digit)
        res = Digit.STREAMS[a.res]

        def setup():
            d.connect()
            d.set_intensity(Digit.LIGHTING_MAX)
            d.set_resolution(res)
            d.set_fps(res["fps"][f"{a.fps}fps"])
            for _ in range(45):  # LED/auto-exposure ramp: brightness settles only after ~40 frames (measured on D21373)
                d.get_frame()
        setup()
        print(f"DIGIT {a.digit}: {a.res} @ {a.fps} fps")

        def rd_digit():
            # digit_interface raises on a single failed read (transient USB drop on Jetson): retry, then reconnect.
            for _ in range(5):
                try:
                    return d.get_frame()
                except Exception:
                    time.sleep(0.05)
            for attempt in range(3):
                print(f"WARNING: DIGIT {a.digit} stopped sending frames -> reconnecting ({attempt + 1}/3)")
                try:
                    d.disconnect()
                except Exception:
                    pass
                time.sleep(1.0)
                try:
                    setup()
                    print("  reconnected. Exposure may differ from before: check the image, run `check` after.")
                    return d.get_frame()
                except Exception:
                    pass
            raise SystemExit(f"DIGIT {a.digit}: no frames after reconnecting; check USB cable/power, then rerun "
                             "(captured files and log are kept).")
        return rd_digit, d.disconnect
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.cam}")

    def rd():
        ok, fr = cap.read()
        return fr if ok else None
    return rd, cap.release


def grab_avg(rd, n):
    frames = [rd().astype(np.float32) for _ in range(n)]
    return np.clip(np.mean(frames, 0) + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- features
def features(img, ref):
    """HxWx5 float32: diff(B,G,R)/255 and X,Y in [-1,1]. Diff (not raw color) per Wedge Fig. 6."""
    H, W = img.shape[:2]
    diff = (cv2.GaussianBlur(img, (3, 3), 0) - ref) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = xx / (W - 1) * 2 - 1
    Y = yy / (H - 1) * 2 - 1
    return np.dstack([diff, X, Y]).astype(np.float32)


# ----------------------------------------------------------------------------- sphere labels
def sphere_grad(H, W, cx, cy, a_px, R_mm, mm_px):
    """Analytic Gx, Gy, depth of a sphere cap with contact radius a_px. Returns (gx, gy, z, mask)."""
    R_px = R_mm / mm_px
    a_px = min(a_px, R_px - 1.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = (xx - cx) * mm_px, (yy - cy) * mm_px
    r2 = dx ** 2 + dy ** 2
    mask = r2 < (a_px * mm_px) ** 2
    s = np.sqrt(np.clip(R_mm ** 2 - r2, 1e-9, None))
    s_a = np.sqrt(R_mm ** 2 - (a_px * mm_px) ** 2)
    gx = np.where(mask, -dx / s, 0.0)
    gy = np.where(mask, -dy / s, 0.0)
    z = np.where(mask, s - s_a, 0.0)
    return gx.astype(np.float32), gy.astype(np.float32), z.astype(np.float32), mask


def ball_r(L, s):
    R = L.get("ball_radius_mm") or s.get("ball_radius_mm")
    if not R:
        raise SystemExit("label entry has no ball_radius_mm")
    return float(R)


def contact_radius(L, s):
    """Contact radius [px]. A known press depth (Z-stage / 3D-printer rig) beats the image fit."""
    if L.get("depth_mm"):
        R, d = ball_r(L, s), L["depth_mm"]
        return float(np.sqrt(R ** 2 - (R - d) ** 2) / s["mm_per_px"])
    return L["r"]


def label_grad(shape, L, s):
    return sphere_grad(*shape, L["cx"], L["cy"], contact_radius(L, s), ball_r(L, s), s["mm_per_px"])


def detect_contact(img, ref, r_max=None):
    """Initial guess only: most circular blob of the diff magnitude within the allowed size."""
    mag = np.linalg.norm(cv2.GaussianBlur(img - ref, (3, 3), 0), axis=2)
    thr = max(3.0 * np.median(np.abs(mag - np.median(mag))) + np.median(mag), 0.35 * mag.max())
    m = (mag > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, score = None, -1
    for c in cnts:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 3 or (r_max and r > r_max):
            continue
        circ = cv2.contourArea(c) / (np.pi * r * r + 1e-6)  # 1 = perfect disc
        sc = circ * np.sqrt(cv2.contourArea(c))
        if sc > score:
            best, score = (float(cx), float(cy), float(r)), sc
    return best


# ----------------------------------------------------------------------------- manual labelling GUI
class CircleLabeler:
    """Left click = set centre, right click = set radius through that point.
    w/a/s/d move 0.25 px, m/n radius +/-0.25 px (UPPERCASE = x4). Enter accept, x skip, u undo-last-accept, Esc quit.
    Inset (top-left) = zoom x6 around the circle."""
    SCALE, ZOOM, HALF = 2, 6, 20

    def __init__(self, title):
        self.title = title
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, self.on_mouse)
        self.c = None

    def on_mouse(self, ev, x, y, flags, _):
        if self.c is None:
            return
        px, py = x / self.SCALE, y / self.SCALE
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.c[0], self.c[1] = px, py
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.c[2] = max(1.0, float(np.hypot(px - self.c[0], py - self.c[1])))

    def draw(self, base, info):
        cx, cy, r = self.c
        S = self.SCALE
        vis = cv2.resize(base, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
        cv2.circle(vis, (int(round(cx * S)), int(round(cy * S))), int(round(r * S)), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(round(cx * S)), int(round(cy * S))), (0, 255, 255), cv2.MARKER_CROSS, 6)
        # zoom inset
        Z, h = self.ZOOM, self.HALF
        x0, y0 = int(round(cx)) - h, int(round(cy)) - h
        pad = cv2.copyMakeBorder(base, h, h, h, h, cv2.BORDER_CONSTANT)
        crop = pad[y0 + h:y0 + 3 * h, x0 + h:x0 + 3 * h]
        ins = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        cv2.circle(ins, (int(round((cx - x0) * Z)), int(round((cy - y0) * Z))), int(round(r * Z)), (0, 255, 255), 1, cv2.LINE_AA)
        ih, iw = ins.shape[:2]
        if ih < vis.shape[0] and iw < vis.shape[1]:
            vis[:ih, :iw] = ins
            cv2.rectangle(vis, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 1)
        cv2.putText(vis, info + f"  r={r:.2f}px", (5, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis

    def run(self, base, init, info):
        self.c = list(init)
        step = {"w": (0, -1, 0), "s": (0, 1, 0), "a": (-1, 0, 0), "d": (1, 0, 0), "m": (0, 0, 1), "n": (0, 0, -1)}
        while True:
            cv2.imshow(self.title, self.draw(base, info))
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            ch = chr(k) if k < 128 else ""
            if k in (13, 10):
                return tuple(self.c)
            if k == 27:
                raise KeyboardInterrupt
            if ch in ("x", "u"):
                return ch
            if ch.lower() in step:
                f = 0.25 * (4 if ch.isupper() else 1)
                dx, dy, dr = step[ch.lower()]
                self.c = [self.c[0] + f * dx, self.c[1] + f * dy, max(1.0, self.c[2] + f * dr)]


def diff_vis(img, ref, gain=3.0):
    return np.clip((img - ref) * gain + 128, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- dataset
def build_samples(data_dir, s, labels, names, bg_frac=0.05, rng=None):
    """Collect (RGBXY, GxGy) pixel samples from labelled ball frames. Wedge: +5% non-contact zeros."""
    rng = rng or np.random.default_rng(0)
    ref = ref_image(os.path.join(data_dir, "ref.png"), s)
    Xs, Ys = [], []
    for n in names:
        L = labels[n]
        F = features(read(os.path.join(data_dir, n), s), ref)
        gx, gy, _, mask = label_grad(F.shape[:2], L, s)
        inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # rim labels least reliable
        Xs.append(F[inner]); Ys.append(np.stack([gx[inner], gy[inner]], 1))
        yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
        far = ((xx - L["cx"]) ** 2 + (yy - L["cy"]) ** 2) > (1.5 * contact_radius(L, s)) ** 2
        idx = np.flatnonzero(far)
        k = max(1, int(bg_frac * inner.sum()))
        pick = rng.choice(idx, size=min(k, idx.size), replace=False)
        Xs.append(F.reshape(-1, 5)[pick]); Ys.append(np.zeros((pick.size, 2), np.float32))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys).astype(np.float32)


def split_names(labels, test_frac=0.2, seed=0):
    names = sorted(labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    k = max(1, int(round(test_frac * len(names))))
    return names[k:], names[:k]  # train, test (split by image, as in the paper: 32/8)


def grad_errors(pred, gt):
    """Table I metrics: |G err| and |atan(G) err| in degrees, for x and y."""
    e = np.abs(pred - gt).mean(0)
    th = np.degrees(np.abs(np.arctan(pred) - np.arctan(gt))).mean(0)
    return {"Gx_err": float(e[0]), "Gy_err": float(e[1]), "thx_deg": float(th[0]), "thy_deg": float(th[1])}


# ----------------------------------------------------------------------------- model (torch, lazy import)
def make_model(hidden=32, act="tanh"):
    import torch.nn as nn
    A = {"tanh": nn.Tanh, "relu": nn.ReLU, "hardtanh": nn.Hardtanh}[act]
    # 1x1 convs == per-pixel MLP 5-32-32-32-2, but runs on a whole image and exports cleanly to NPUs
    return nn.Sequential(
        nn.Conv2d(5, hidden, 1), A(),
        nn.Conv2d(hidden, hidden, 1), A(),
        nn.Conv2d(hidden, hidden, 1), A(),
        nn.Conv2d(hidden, 2, 1),
    )


def fit_mlp(Xtr, Ytr, Xte, Yte, act="tanh", epochs=300, lr=1e-3, batch=4096, seed=0):
    import torch
    torch.manual_seed(seed)
    model = make_model(act=act)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.from_numpy(Xtr)[:, :, None, None]; Yt = torch.from_numpy(Ytr)[:, :, None, None]
    Xv = torch.from_numpy(Xte)[:, :, None, None]; Yv = torch.from_numpy(Yte)[:, :, None, None]
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            loss = torch.nn.functional.mse_loss(model(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.nn.functional.l1_loss(model(Xv), Yv).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"ep {ep:4d}  train_mse {loss.item():.5f}  test_l1 {v:.5f}")
    model.load_state_dict(best_state)
    return model


def torch_predictor(model_path):
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = make_model(act=ck["act"]); model.load_state_dict(ck["state"]); model.eval()

    def predict(F):  # HxWx5 -> HxWx2
        with torch.no_grad():
            x = torch.from_numpy(F).permute(2, 0, 1)[None]
            return model(x)[0].permute(1, 2, 0).numpy()
    return predict


# ----------------------------------------------------------------------------- Poisson + reconstruction
def poisson_dst(gx, gy):
    """Fast Poisson solve (Dirichlet 0 boundary), DST-I. Returns z in pixel units of the gradients."""
    H, W = gx.shape
    gxx = np.zeros_like(gx, dtype=np.float64); gyy = np.zeros_like(gy, dtype=np.float64)
    gxx[:, 1:] = gx[:, 1:] - gx[:, :-1]
    gyy[1:, :] = gy[1:, :] - gy[:-1, :]
    f = (gxx + gyy)[1:-1, 1:-1]
    h, w = f.shape
    lam = (2 * np.cos(np.pi * np.arange(1, w + 1) / (w + 1)) - 2)[None, :] + \
          (2 * np.cos(np.pi * np.arange(1, h + 1) / (h + 1)) - 2)[:, None]
    z = np.zeros((H, W))
    z[1:-1, 1:-1] = idstn(dstn(f, type=1) / lam, type=1)
    return z


def nearest_fill(g, mask):
    """Replace gradients under mask (e.g. markers) by nearest valid value (Wedge: 'nearest', ~10 ms)."""
    if mask is None or not mask.any():
        return g
    _, (iy, ix) = distance_transform_edt(mask, return_indices=True)
    return g[iy, ix]


def marker_mask(img, dilate=2):
    """Dark dots -> mask. Tune thresholds for your marker ink."""
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (25, 25), 0)
    m = ((bg - gray) > 25).astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate).astype(bool)


def reconstruct(img, ref, predict, s, markers=False, contact_gate=0.0):
    """img/ref: prepped BGR float. Returns depth_mm (HxW), grad (HxWx2)."""
    F = features(img, ref)
    G = predict(F)
    if contact_gate > 0:  # optional: zero gradients where diff is tiny (suppresses drift/noise)
        G[np.linalg.norm(F[..., :3], axis=2) < contact_gate] = 0
    if markers:
        G = nearest_fill(G, marker_mask(img))
    depth = poisson_dst(G[..., 0], G[..., 1]) * s["mm_per_px"]
    return depth.astype(np.float32), G


def depth_vis(depth, vmax=None):
    vmax = vmax or max(1e-6, float(np.percentile(depth, 99.9)))
    return cv2.applyColorMap(np.clip(depth / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


# ----------------------------------------------------------------------------- synthetic sensor
def render(z, s, rng, noise=1.0):
    """Lambertian gel, 3 coloured side lights (Wedge-like: R left, G bottom, B right), attenuation."""
    H, W = z.shape
    zpx = z / s["mm_per_px"]
    gy, gx = np.gradient(cv2.GaussianBlur(zpx, (5, 5), 1.0))
    n = np.dstack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H - 1, W - 1])[:, None, None]
    el = np.radians(25)
    lights = [  # (direction towards light, BGR colour, attenuation map)
        ((-np.cos(el), 0, np.sin(el)), (40, 40, 220), np.exp(-xx / 0.9)),
        ((0, np.cos(el), np.sin(el)), (40, 200, 40), np.exp(-(1 - yy) / 0.9)),
        ((np.cos(el), 0, np.sin(el)), (220, 60, 40), np.exp(-(1 - xx) / 0.9)),
    ]
    img = np.zeros((H, W, 3))
    for d, c, att in lights:
        shade = np.clip(n @ np.array(d), 0, None)
        img += shade[..., None] * att[..., None] * np.array(c)[None, None, :]
    img = img * 0.9 + 30 + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def cmd_synth(a):
    s = load_sensor(a.sensor); rng = np.random.default_rng(a.seed)
    H, W = s["height"], s["width"]
    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(os.path.join(a.out, "ref.png"), render(np.zeros((H, W)), s, rng))
    gt = {}
    for i in range(a.n):
        R = float(rng.choice(a.ball_r))
        d = rng.uniform(0.15, 0.35) * R
        a_px = np.sqrt(R ** 2 - (R - d) ** 2) / s["mm_per_px"]
        cx, cy = rng.uniform(a_px + 5, W - a_px - 5), rng.uniform(a_px + 5, H - a_px - 5)
        _, _, z, _ = sphere_grad(H, W, cx, cy, a_px, R, s["mm_per_px"])
        name = f"ball_{i:03d}.png"
        cv2.imwrite(os.path.join(a.out, name), render(z, s, rng))
        gt[name] = {"cx": cx, "cy": cy, "r": float(a_px), "ball_radius_mm": R, "depth_mm": float(d)}
    json.dump(gt, open(os.path.join(a.out, "gt_synth.json"), "w"), indent=1)
    print(f"wrote {a.n} frames + ref.png to {a.out} (balls {a.ball_r} mm)")


def cmd_capture(a):
    rd, close = open_source(a)
    os.makedirs(a.out, exist_ok=True)
    log_p = os.path.join(a.out, "capture_log.json")
    log = json.load(open(log_p)) if os.path.exists(log_p) else []
    i = len(glob.glob(os.path.join(a.out, "ball_*.png")))
    have_ref = os.path.exists(os.path.join(a.out, "ref.png"))
    print(f"ball radius for these presses: {a.ball_r} mm (change with --ball-r, restart when you swap balls)")
    print("r = ref (avg) FIRST | space = ball frame | e = end ref (avg) | q = quit")
    shape = None
    try:
        while True:
            fr = rd()
            if fr is None:
                break
            if shape is None:
                shape = list(fr.shape); log.append({"frame_shape": shape}); print(f"frame H x W x C = {fr.shape}")
            vis = fr.copy()
            cv2.putText(vis, f"n={i} {'REF OK' if have_ref else 'NO REF: press r'}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if have_ref else (0, 0, 255), 1)
            cv2.imshow("capture", vis)
            k = cv2.waitKey(1) & 0xFF
            t = time.strftime("%Y-%m-%d %H:%M:%S")
            if k in (ord("r"), ord("e")):
                name = "ref.png" if k == ord("r") else "ref_end.png"
                cv2.imwrite(os.path.join(a.out, name), grab_avg(rd, a.n_avg))
                log.append({"file": name, "time": t}); have_ref |= k == ord("r"); print(f"{name} saved ({a.n_avg} avg)")
            elif k == ord(" "):
                if not have_ref:
                    print("take the ref first (r)"); continue
                name = f"ball_{i:03d}.png"
                cv2.imwrite(os.path.join(a.out, name), fr)
                log.append({"file": name, "time": t, "ball_radius_mm": a.ball_r}); print(name); i += 1
            elif k == ord("q"):
                break
    finally:
        json.dump(log, open(log_p, "w"), indent=1)
        close(); cv2.destroyAllWindows()


def cmd_measure(a):
    """Click A then B on a known distance (caliper jaws, ruler ticks, pin pitch); type the distance in mm."""
    s = load_sensor(a.sensor, need_scale=False)
    rd = close = None
    if a.img:
        base = prep(cv2.imread(a.img), s).astype(np.uint8)
        frozen = True
    else:  # live view; space = freeze (avg of 5 frames) for clicking, space again = back to live
        rd, close = open_source(a); base = None
        frozen = False
    S = a.scale
    pts, res = [], []
    title = "measure: [live] space=freeze | [frozen] click A, B, type mm in terminal; space=live | q=finish"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and frozen:
            pts.append((x / S, y / S))
    cv2.namedWindow(title); cv2.setMouseCallback(title, on_mouse)
    while True:
        if not frozen:
            fr = rd()
            if fr is not None:
                base = prep(fr, s).astype(np.uint8)
        if base is None:
            cv2.waitKey(20); continue
        vis = cv2.resize(base, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, "FROZEN" if frozen else "LIVE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if frozen else (0, 255, 0), 2)
        for p in pts:
            cv2.drawMarker(vis, (int(p[0] * S), int(p[1] * S)), (0, 255, 255), cv2.MARKER_CROSS, 12)
        cv2.imshow(title, vis)
        k = cv2.waitKey(1 if not frozen else 20) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and rd is not None:
            frozen = not frozen
            pts.clear()
            if frozen:
                base = prep(grab_avg(rd, 5), s).astype(np.uint8)
            continue
        if len(pts) == 2:
            cv2.imshow(title, vis); cv2.waitKey(1)
            (x0, y0), (x1, y1) = pts
            dpx = float(np.hypot(x1 - x0, y1 - y0))
            mm = input(f"{dpx:.2f} px -> distance in mm (empty = discard): ").strip()
            if mm:
                ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180
                res.append({"mm": float(mm), "px": dpx, "mm_per_px": float(mm) / dpx,
                            "axis": "x" if ang < 45 or ang > 135 else "y"})
                print(f"  -> {res[-1]['mm_per_px']:.5f} mm/px ({res[-1]['axis']})")
            pts.clear()
    cv2.destroyAllWindows()
    if close is not None:
        close()
    if not res:
        return
    v = np.array([r["mm_per_px"] for r in res])
    print(f"\nN={len(v)}  mean {v.mean():.5f} mm/px  std {v.std():.5f}  ({100 * v.std() / v.mean():.1f}%)")
    for ax in ("x", "y"):
        w = [r["mm_per_px"] for r in res if r["axis"] == ax]
        if w:
            print(f"  axis {ax}: N={len(w)} mean {np.mean(w):.5f}")
    json.dump(res, open(a.log, "w"), indent=1)
    if a.write:
        s["mm_per_px"] = round(float(v.mean()), 5)
        json.dump(s, open(a.sensor, "w"), indent=1)
        print(f"mm_per_px written to {a.sensor}")


def cmd_check(a):
    """Background drift + light-leak indicators. Run before labelling."""
    s = load_sensor(a.sensor, need_scale=False)
    ref = read(os.path.join(a.data, "ref.png"), s)
    print(f"{'file':14s} offset B/G/R      changed>{a.thr:.0f}DN")
    for p in sorted(glob.glob(os.path.join(a.data, "ball_*.png"))) + glob.glob(os.path.join(a.data, "ref_end.png")):
        d = read(p, s) - ref
        med = np.median(d.reshape(-1, 3), 0)
        big = (np.linalg.norm(d, axis=2) > a.thr).mean() * 100
        flag = "  <-- drift" if np.abs(med).max() > 2 else ""
        flag += "  <-- large area (leak/shadow?)" if big > a.leak and "ball_" in p else ""
        print(f"{os.path.basename(p):14s} {med.round(1)!s:18s} {big:5.1f}%{flag}")
    print("expect: |offset| <= ~2 DN; changed ~ contact area only (a few %). ref_end = drift over the session.")


def cmd_label(a):
    s = load_sensor(a.sensor)
    ref = ref_image(os.path.join(a.data, "ref.png"), s)
    out = os.path.join(a.data, "labels.json")
    labels = json.load(open(out)) if os.path.exists(out) else {}
    log_p = os.path.join(a.data, "capture_log.json")
    log_r = {e["file"]: e.get("ball_radius_mm") for e in json.load(open(log_p)) if "file" in e} if os.path.exists(log_p) else {}
    files = sorted(glob.glob(os.path.join(a.data, "ball_*.png")))
    todo = [p for p in files if a.redo or os.path.basename(p) not in labels]
    gui = CircleLabeler("label: L-click centre, R-click rim, wasd/mn fine, Enter ok, x skip, u undo, Esc quit")
    i, history = 0, []
    while i < len(todo):
        n = os.path.basename(todo[i])
        R = a.ball_r or log_r.get(n) or (labels.get(n) or {}).get("ball_radius_mm")
        if not R:
            raise SystemExit(f"{n}: unknown ball radius. Pass --ball-r or capture with `capture --ball-r`.")
        R_px = R / s["mm_per_px"]
        img = read(todo[i], s)
        init = labels.get(n) or detect_contact(img, ref, r_max=R_px)
        init = (init["cx"], init["cy"], init["r"]) if isinstance(init, dict) else init or (img.shape[1] / 2, img.shape[0] / 2, R_px / 2)
        try:
            c = gui.run(diff_vis(img, ref, a.gain), init, f"{n} [{i + 1}/{len(todo)}] ball R={R}mm max r={R_px:.1f}px")
        except KeyboardInterrupt:
            break
        if c == "u" and history:
            labels.pop(history.pop(), None); i -= 1; continue
        if c in ("x", "u"):
            i += 1; continue
        if c[2] >= R_px:
            print(f"{n}: r={c[2]:.1f}px >= ball radius {R_px:.1f}px -> check mm_per_px / ball size; not saved"); continue
        labels[n] = {"cx": round(c[0], 2), "cy": round(c[1], 2), "r": round(c[2], 2), "ball_radius_mm": R}
        history.append(n); i += 1
        json.dump(labels, open(out, "w"), indent=1)  # save after every image
    cv2.destroyAllWindows()
    json.dump(labels, open(out, "w"), indent=1)
    print(f"{len(labels)} labels -> {out}")


def cmd_review(a):
    s = load_sensor(a.sensor)
    ref = ref_image(os.path.join(a.data, "ref.png"), s)
    labels = json.load(open(os.path.join(a.data, "labels.json")))
    H, W = ref.shape[:2]  # = sensor width/height after prep; also works when they are null in the json
    tiles, cov = [], np.zeros((H, W, 3), np.uint8)
    h = int(max(contact_radius(L, s) for L in labels.values()) * 1.3) + 4  # crop half-size fits the largest contact
    Z = 192 / (2 * h)                                                    # every tile 192x192 px
    for n in sorted(labels):
        L = labels[n]; r = contact_radius(L, s)
        v = diff_vis(read(os.path.join(a.data, n), s), ref, 3.0)
        pad = cv2.copyMakeBorder(v, h, h, h, h, cv2.BORDER_CONSTANT)
        x0, y0 = int(round(L["cx"])), int(round(L["cy"]))
        t = cv2.resize(pad[y0:y0 + 2 * h, x0:x0 + 2 * h], (192, 192), interpolation=cv2.INTER_NEAREST)
        cv2.circle(t, (int((L["cx"] - x0 + h) * Z), int((L["cy"] - y0 + h) * Z)), int(r * Z), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(t, f"{n[5:8]} r{r:.1f} R{L['ball_radius_mm']}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t)
        cv2.circle(cov, (x0, y0), max(1, int(r)), (0, 200, 255), 1)
        edge = min(L["cx"], L["cy"], W - L["cx"], H - L["cy"]) - r
        if edge < 5:
            print(f"{n}: contact within {edge:.1f}px of the border -> consider dropping")
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(a.data, "labels_review.png"), sheet)
    cv2.imwrite(os.path.join(a.data, "labels_coverage.png"), cv2.resize(cov, None, fx=2, fy=2))
    print(f"wrote labels_review.png ({len(labels)} labels) and labels_coverage.png")


def cmd_train(a):
    s = load_sensor(a.sensor)
    labels = json.load(open(os.path.join(a.data, "labels.json")))
    tr, te = split_names(labels, a.test_frac, a.seed)
    Xtr, Ytr = build_samples(a.data, s, labels, tr)
    Xte, Yte = build_samples(a.data, s, labels, te)
    print(f"train imgs {len(tr)} ({len(Xtr)} px)  test imgs {len(te)} ({len(Xte)} px)")
    model = fit_mlp(Xtr, Ytr, Xte, Yte, act=a.act, epochs=a.epochs)
    import torch
    with torch.no_grad():
        P = model(torch.from_numpy(Xte)[:, :, None, None])[:, :, 0, 0].numpy()
    contact = np.any(Yte != 0, 1)
    err = grad_errors(P[contact], Yte[contact])
    print("test (contact px):", json.dumps(err, indent=1))
    torch.save({"state": model.state_dict(), "act": a.act, "test_names": te, "test_grad_err": err}, a.out)
    print(f"saved {a.out}")


def cmd_recon(a):
    s = load_sensor(a.sensor)
    predict = torch_predictor(a.model)
    depth, _ = reconstruct(read(a.img, s), ref_image(a.ref, s), predict, s, markers=a.markers, contact_gate=a.gate)
    stem = os.path.splitext(a.out)[0]
    np.save(stem + ".npy", depth)
    cv2.imwrite(stem + ".png", depth_vis(depth))
    print(f"max depth {depth.max():.3f} mm, min {depth.min():.3f} mm -> {stem}.npy/.png")


def eval_depth(data_dir, s, labels, names, predict, **kw):
    ref = ref_image(os.path.join(data_dir, "ref.png"), s)
    rows = []
    for n in names:
        L = labels[n]
        depth, _ = reconstruct(read(os.path.join(data_dir, n), s), ref, predict, s, **kw)
        _, _, zgt, mask = label_grad(depth.shape, L, s)
        rows.append({"img": n, "ball_mm": ball_r(L, s), "peak_gt_mm": float(zgt.max()),
                     "peak_pred_mm": float(depth[mask].max()),
                     "rmse_contact_mm": float(np.sqrt(np.mean((depth[mask] - zgt[mask]) ** 2))),
                     "min_depth_mm": float(depth.min())})  # negative = Poisson artefact
    return rows


def cmd_eval(a):
    s = load_sensor(a.sensor)
    import torch
    labels = json.load(open(os.path.join(a.data, "labels.json")))
    names = torch.load(a.model, map_location="cpu", weights_only=False)["test_names"]
    rows = eval_depth(a.data, s, labels, names, torch_predictor(a.model), markers=a.markers, contact_gate=a.gate)
    os.makedirs(a.results, exist_ok=True)
    json.dump(rows, open(os.path.join(a.results, "eval.json"), "w"), indent=1)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("mean RMSE (mm):", round(float(np.mean([r["rmse_contact_mm"] for r in rows])), 4),
          "| mean peak error (mm):", round(float(np.mean([r["peak_pred_mm"] - r["peak_gt_mm"] for r in rows])), 4))


def cmd_live(a):
    s = load_sensor(a.sensor)
    predict = torch_predictor(a.model)
    rd, close = open_source(a)
    ref = None
    print("r = re-take reference (no contact!) | q = quit")
    try:
        while True:
            fr = rd()
            if fr is None:
                break
            if ref is None:
                ref = cv2.GaussianBlur(prep(grab_avg(rd, 10), s), (3, 3), 0); continue
            img = prep(fr, s)
            depth, _ = reconstruct(img, ref, predict, s, markers=a.markers, contact_gate=a.gate)
            cv2.imshow("img | depth", np.hstack([img.astype(np.uint8), depth_vis(depth, a.vmax)]))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("r"):
                ref = None
            elif k == ord("q"):
                break
    finally:
        close(); cv2.destroyAllWindows()


def cmd_export(a):
    import torch
    s = load_sensor(a.sensor)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = make_model(act=ck["act"]); model.load_state_dict(ck["state"]); model.eval()
    if not s.get("width") or not s.get("height"):
        raise SystemExit("export needs width/height in the sensor json (fixed input size for the NPU)")
    dummy = torch.zeros(1, 5, s["height"], s["width"])
    torch.onnx.export(model, dummy, a.out, input_names=["rgbxy"], output_names=["gxgy"], opset_version=13)
    print(f"ONNX -> {a.out}  (input NCHW 1x5x{s['height']}x{s['width']}; Poisson stays on CPU)")


def add_source(q):
    q.add_argument("--cam", type=int, default=0, help="OpenCV camera index (DIGIT is also a UVC camera)")
    q.add_argument("--digit", default=None, help="DIGIT serial, e.g. D20001 (uses digit-interface)")
    q.add_argument("--res", default="QVGA", choices=["QVGA", "VGA"])
    q.add_argument("--fps", type=int, default=30)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", default="sensor.json")
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("synth"); q.add_argument("--out", default="data_synth"); q.add_argument("--n", type=int, default=40)
    q.add_argument("--ball-r", type=float, nargs="+", default=[2.0, 3.0]); q.add_argument("--seed", type=int, default=0)
    q = sp.add_parser("capture"); add_source(q); q.add_argument("--out", default="data")
    q.add_argument("--ball-r", type=float, required=True, help="radius (mm) of the ball used in this session")
    q.add_argument("--n-avg", type=int, default=10)
    q = sp.add_parser("measure"); add_source(q); q.add_argument("--img", default=None, help="image of the gauge; omit to grab from camera")
    q.add_argument("--scale", type=int, default=3); q.add_argument("--write", action="store_true"); q.add_argument("--log", default="measure_log.json")
    q = sp.add_parser("check"); q.add_argument("--data", default="data"); q.add_argument("--thr", type=float, default=15)
    q.add_argument("--leak", type=float, default=8, help="flag frames whose changed area exceeds this %%")
    q = sp.add_parser("label"); q.add_argument("--data", default="data"); q.add_argument("--redo", action="store_true")
    q.add_argument("--ball-r", type=float, default=None, help="override ball radius (mm) for all images")
    q.add_argument("--gain", type=float, default=3.0, help="diff display gain")
    q = sp.add_parser("review"); q.add_argument("--data", default="data")
    q = sp.add_parser("train"); q.add_argument("--data", default="data"); q.add_argument("--out", default="wedge_mlp.pt")
    q.add_argument("--act", default="tanh", choices=["tanh", "relu", "hardtanh"]); q.add_argument("--epochs", type=int, default=300)
    q.add_argument("--test-frac", type=float, default=0.2); q.add_argument("--seed", type=int, default=0)
    for name in ("recon", "eval", "live"):
        q = sp.add_parser(name); q.add_argument("--model", default="wedge_mlp.pt"); q.add_argument("--markers", action="store_true")
        q.add_argument("--gate", type=float, default=0.0, help="zero grads where |diff|/255 < gate (e.g. 0.02)")
        if name == "recon":
            q.add_argument("--ref", required=True); q.add_argument("--img", required=True); q.add_argument("--out", default="depth.png")
        if name == "eval":
            q.add_argument("--data", default="data"); q.add_argument("--results", default="results")
        if name == "live":
            add_source(q); q.add_argument("--vmax", type=float, default=None)
    q = sp.add_parser("export"); q.add_argument("--model", default="wedge_mlp.pt"); q.add_argument("--out", default="wedge_mlp.onnx")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
