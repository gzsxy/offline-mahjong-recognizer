#!/usr/bin/env python3
"""Synthetic training data generator for mahjong tile detection (YOLOv8).

Generates 640x640 images with scattered mahjong tiles matching the target
inference distribution (tile long edge ~60-170px in 640px crops).

Classes: 0-8 wan1..wan9, 9-17 tong1..tong9, 18-26 tiao1..tiao9, 27 back.
Fixed seed -> fully reproducible.

Boost mode (--back-range LO,HI): force a per-image fraction of tiles to be
backs, mixing real back sprites / FluffyStuff Back.png / procedural backs.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]

TILE_DIRS = [
    ROOT / "training/assets/tiles-fluffystuff/Export/Regular",
    ROOT / "training/assets/tiles-fluffystuff/Export/Black",
]
BACK_SPRITES_DIR = ROOT / "training/assets/back-sprites"

CLASS_NAMES = (
    [f"wan{i}" for i in range(1, 10)]
    + [f"tong{i}" for i in range(1, 10)]
    + [f"tiao{i}" for i in range(1, 10)]
    + ["back"]
)
NC = len(CLASS_NAMES)  # 28

# class_id -> asset filename stem (0-26 faces; 27 = back)
FACE_FILES = {}
for i in range(9):
    FACE_FILES[i] = f"Man{i+1}"      # wan
    FACE_FILES[9 + i] = f"Pin{i+1}"  # tong
    FACE_FILES[18 + i] = f"Sou{i+1}"  # tiao
BACK_ID = 27

CANVAS = 640
TILE_LONG_RANGE = (60.0, 170.0)  # 牌面长边像素范围，随 --canvas 缩放
TILE_ASPECT = 600 / 800  # w/h of source art


# ---------------------------------------------------------------- tiles ----
def load_tiles():
    """Return {class_id: [PIL RGBA images per variant], 'back_sprites': [...]}."""
    tiles = {}
    for cid, stem in FACE_FILES.items():
        tiles[cid] = [Image.open(d / f"{stem}.png").convert("RGBA") for d in TILE_DIRS]
    back_paths = [d / "Back.png" for d in TILE_DIRS]
    if all(p.exists() for p in back_paths):
        kept = [im for im in
                (Image.open(p).convert("RGBA") for p in back_paths)
                if _is_blue_green_back(im)]
        tiles[BACK_ID] = kept or None  # 全被过滤则回退程序化生成
    else:
        tiles[BACK_ID] = None  # procedural fallback
    sprites = []
    if BACK_SPRITES_DIR.exists():
        for p in sorted(BACK_SPRITES_DIR.glob("*.png")):
            sprites.append(Image.open(p).convert("RGBA"))
    # 背精灵同样限定蓝/绿（剔除黑背变体与公开集杂色背）
    tiles["back_sprites"] = [im for im in sprites if _is_blue_green_back(im)]
    return tiles


def _is_blue_green_back(im, min_ratio=0.18):
    """蓝/绿背判定：与 App 端 hasColoredBackAppearance 同口径。"""
    arr = np.asarray(im.convert("RGB").resize((80, 100)), dtype=np.int32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    spread = mx - mn
    colored = (mx > 0) & (spread > 25) & (spread * 5 > mx)
    blue_back = (b * 100 > r * 112) & (b * 100 > g * 103)
    green_back = (g * 100 > r * 112) & (g * 100 > b * 103)
    hit = colored & (blue_back | green_back)
    return bool(hit.mean() >= min_ratio)


def procedural_back(rng):
    """Draw a tile back: rounded rect, colored base, texture."""
    w, h = 600, 800
    families = [
        (30, 90, 200),   # blue
        (30, 140, 90),   # green
        (20, 140, 160),  # teal
        (40, 120, 180),  # steel blue
        (50, 160, 120),  # jade
    ]
    # 牌背限定蓝/绿系：黑背/杂色背不在目标分布内（2026-09-08 需求）
    base = rng.choice(families)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, w - 5, h - 5], radius=60, fill=base + (255,))
    # diagonal stripes
    stripe = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stripe)
    for x in range(-h, w, 46):
        sd.line([(x, 0), (x + h, h)], fill=(255, 255, 255, 14), width=10)
    img = Image.alpha_composite(img, stripe)
    # noise
    arr = np.array(img).astype(np.int16)
    nrng = np.random.default_rng(rng.randrange(2**31))
    arr[..., :3] += nrng.normal(0, 6, arr[..., :3].shape).astype(np.int16)
    arr[..., :3] = np.clip(arr[..., :3], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    # light edge
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([14, 14, w - 15, h - 15], radius=50,
                        outline=(255, 255, 255, 70), width=10)
    return img


def render_tile(rng, tiles, forced_cid=None, back_prob=0.08):
    """Pick class/variant/scale/rotation. Returns (sprite RGBA, class_id)."""
    if forced_cid is not None:
        cid = forced_cid
    elif rng.random() < back_prob:
        cid = BACK_ID
    else:
        cid = rng.randint(0, 26)
    if cid == BACK_ID:
        sprites = tiles.get("back_sprites") or []
        r = rng.random()
        if sprites and r < 0.40:
            src = rng.choice(sprites)            # real back crop
        elif tiles[BACK_ID] is not None and r < 0.70:
            src = rng.choice(tiles[BACK_ID])     # FluffyStuff Back.png
        else:
            src = procedural_back(rng)           # procedural
    else:
        src = rng.choice(tiles[cid])
    long_side = rng.uniform(*TILE_LONG_RANGE)
    w = max(8, int(round(long_side * TILE_ASPECT)))
    h = int(round(long_side))
    spr = src.resize((w, h), Image.BILINEAR)
    angle = rng.uniform(0, 360)
    spr = spr.rotate(angle, expand=True, resample=Image.BICUBIC)
    return spr, cid


# ------------------------------------------------------------ background ----
def _finalize_bg(arr, rng, nrng):
    """Lighting gradient + vignette, arr float32 HxWx3."""
    h, w = arr.shape[:2]
    # global lighting ramp in a random direction
    ang = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(-0.12, 0.12)
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    ramp = 1.0 + strength * (math.cos(ang) * xs + math.sin(ang) * ys) / 1.414
    arr *= ramp[..., None]
    # vignette
    r2 = (xs ** 2 + ys ** 2) / 2.0
    v = rng.uniform(0.15, 0.35)
    arr *= (1.0 - v * r2)[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_background(rng, nrng):
    h = w = CANVAS
    kind = rng.choice(["felt", "wood", "solid"])
    if kind == "felt":
        fams = [(52, 110, 60), (50, 80, 120), (110, 110, 110), (150, 125, 85)]
        base = np.array(rng.choice(fams), dtype=np.float32)
        base += nrng.normal(0, 8, 3).astype(np.float32)
        arr = np.full((h, w, 3), base, dtype=np.float32)
        # coarse mottling upscaled
        coarse = nrng.normal(0, 1, (40, 40, 1)).astype(np.float32)
        coarse = np.array(Image.fromarray(
            ((coarse[..., 0] - coarse.min()) / (np.ptp(coarse) + 1e-6) * 255)
            .astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32)
        arr += ((coarse - 127.5) / 127.5 * 9)[..., None]
        arr += nrng.normal(0, rng.uniform(5, 10), (h, w, 3)).astype(np.float32)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
        arr = np.array(img, dtype=np.float32)
    elif kind == "wood":
        hue = rng.uniform(20, 40)  # brown-ish
        base_val = rng.uniform(90, 170)
        y = np.arange(h, dtype=np.float32)[:, None]
        x = np.arange(w, dtype=np.float32)[None, :]
        lam1 = rng.uniform(40, 120)
        lam2 = rng.uniform(8, 25)
        val = (base_val
               + rng.uniform(6, 16) * np.sin(2 * np.pi * y / lam1 + rng.uniform(0, 6.28))
               + rng.uniform(2, 6) * np.sin(2 * np.pi * y / lam2 + rng.uniform(0, 6.28))
               + rng.uniform(1, 4) * np.sin(2 * np.pi * x / rng.uniform(150, 400)
                                            + y * rng.uniform(0.01, 0.05)))
        # plank boundaries (horizontal)
        n_pl = rng.randint(2, 5)
        bounds = sorted(rng.uniform(0, h) for _ in range(n_pl))
        plank_adj = np.zeros((h, 1), dtype=np.float32)
        cur = 0.0
        bi = 0
        adj = rng.uniform(-14, 14)
        for yy in range(h):
            while bi < len(bounds) and yy >= bounds[bi]:
                adj = rng.uniform(-14, 14)
                bi += 1
            plank_adj[yy, 0] = adj
        val += plank_adj
        for b in bounds:  # dark seam lines
            y0 = int(b)
            val[max(0, y0 - 1):y0 + 1, :] -= rng.uniform(15, 30)
        # convert HSV-ish: brown = (hue, high sat)
        import colorsys
        r0, g0, b0 = colorsys.hsv_to_rgb(hue / 360.0, rng.uniform(0.35, 0.6), 1.0)
        tint = np.array([r0, g0, b0], dtype=np.float32)
        arr = val[..., None] * tint[None, None, :]
        arr += nrng.normal(0, 3.5, (h, w, 3)).astype(np.float32)
    else:  # solid with color-temperature gradient
        fams = [(128, 128, 128), (190, 180, 160), (170, 140, 105), (200, 195, 185)]
        base = np.array(rng.choice(fams), dtype=np.float32)
        base += nrng.normal(0, 6, 3).astype(np.float32)
        arr = np.full((h, w, 3), base, dtype=np.float32)
        ang = rng.uniform(0, 2 * math.pi)
        xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        g = (math.cos(ang) * xs + math.sin(ang) * ys)
        t = rng.uniform(4, 12)
        arr[..., 0] += t * g   # warm
        arr[..., 2] -= t * g   # cool
        arr += nrng.normal(0, rng.uniform(2, 4), (h, w, 3)).astype(np.float32)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(1.0))
        arr = np.array(img, dtype=np.float32)
    return Image.fromarray(_finalize_bg(arr, rng, nrng))


# ------------------------------------------------------------ geometry ----
def rect_inter(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def rect_area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def union_area(rects):
    """Exact union area of a small list of rects via coordinate compression."""
    if not rects:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            for r in rects:
                if r[0] <= cx < r[2] and r[1] <= cy < r[3]:
                    area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
                    break
    return area


# ------------------------------------------------------------- compose -----
def generate_image(seed, tiles, cfg):
    """Returns (PIL RGB image, list of (class_id, cx, cy, w, h) normalized)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed & 0x7FFFFFFF)
    img = make_background(rng, nrng).convert("RGBA")

    if rng.random() < cfg.empty_prob:
        n_tiles = 0
    else:
        n_tiles = rng.randint(cfg.tiles_min, cfg.tiles_max)

    # forced-back plan (boost mode)
    forced = [None] * n_tiles
    if n_tiles and cfg.back_range:
        lo, hi = cfg.back_range
        k = max(1, int(round(n_tiles * rng.uniform(lo, hi))))
        for i in rng.sample(range(n_tiles), min(k, n_tiles)):
            forced[i] = BACK_ID

    # per-image shadow direction (single light source), 随画布等比缩放
    sh_ang = math.radians(rng.uniform(45, 135))
    scale = CANVAS / 640.0
    if rng.random() < getattr(cfg, "strong_shadow_prob", 0.0):
        # 强阴影场景：低角度强光，长而深的投影（640 基准 9-21px）
        sh_mag = rng.uniform(9, 21) * scale
        sh_dark = rng.uniform(0.70, 0.90)
        sh_dilate, sh_blur = 11, rng.uniform(3.0, 6.0)
    else:
        sh_mag = rng.uniform(2, 4) * scale
        sh_dark = 0.55
        sh_dilate, sh_blur = 5, 1.5
    sh_dx, sh_dy = sh_mag * math.cos(sh_ang), sh_mag * math.sin(sh_ang)

    placed = []  # dicts: aabb(full), sprite, cid
    for t in range(n_tiles):
        spr, cid = render_tile(rng, tiles, forced_cid=forced[t],
                               back_prob=cfg.back_prob)
        sw, sh = spr.size
        best = None
        for _try in range(60):
            cx = rng.uniform(0, CANVAS)
            cy = rng.uniform(0, CANVAS)
            aabb = (cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2)
            canvas_r = (0, 0, CANVAS, CANVAS)
            vis = rect_inter(aabb, canvas_r)
            if vis is None:
                continue
            full_area = rect_area(aabb)
            if rect_area(vis) < 0.8 * full_area:  # clip <= 20%
                continue
            ov = sum(rect_area(rect_inter(aabb, p["aabb"]) or (0, 0, 0, 0))
                     for p in placed)
            if ov <= 0.30 * full_area:
                best = (cx, cy)
                break
        if best is None:
            continue  # could not place without too much overlap; skip tile
        placed.append({"aabb": (best[0] - sw / 2, best[1] - sh / 2,
                                best[0] + sw / 2, best[1] + sh / 2),
                       "cx": best[0], "cy": best[1], "spr": spr, "cid": cid})

    # paste in placement order (later tiles occlude earlier ones)
    for p in placed:
        spr = p["spr"]
        x0 = int(round(p["cx"] - spr.size[0] / 2))
        y0 = int(round(p["cy"] - spr.size[1] / 2))
        # thickness/shadow: dark, slightly dilated silhouette at offset
        alpha = spr.getchannel("A")
        sil = Image.new("RGBA", spr.size, (10, 10, 15, 0))
        sil.putalpha(alpha.point(lambda a: int(a * sh_dark)))
        sil = sil.filter(ImageFilter.MaxFilter(sh_dilate)).filter(
            ImageFilter.GaussianBlur(sh_blur))
        img.alpha_composite(sil, (int(round(x0 + sh_dx)), int(round(y0 + sh_dy))))
        img.alpha_composite(spr, (x0, y0))

    # labels: visibility = canvas-visible area minus occlusion by later tiles
    labels = []
    canvas_r = (0, 0, CANVAS, CANVAS)
    for i, p in enumerate(placed):
        aabb = p["aabb"]
        vis_r = rect_inter(aabb, canvas_r)
        base_vis = rect_area(vis_r)
        later = [q["aabb"] for q in placed[i + 1:]]
        occ_rects = []
        for b in later:
            inter = rect_inter(vis_r, b)
            if inter:
                occ_rects.append(inter)
        visible = base_vis - union_area(occ_rects)
        if visible < 0.5 * rect_area(aabb):
            continue
        cx = (vis_r[0] + vis_r[2]) / 2 / CANVAS
        cy = (vis_r[1] + vis_r[3]) / 2 / CANVAS
        bw = (vis_r[2] - vis_r[0]) / CANVAS
        bh = (vis_r[3] - vis_r[1]) / CANVAS
        labels.append((p["cid"], cx, cy, bw, bh))

    img = img.convert("RGB")
    if rng.random() < getattr(cfg, "persp_prob", 0.0):
        img, labels = apply_perspective(img, labels, rng)
    img = post_process(img, rng, nrng)
    if rng.random() < getattr(cfg, "glare_prob", 0.0):
        img = add_glare(img, rng)
    return img, labels


# -------------------------------------------------- hard scenes (v2) -------
def solve_homography(src_pts, dst_pts):
    """DLT 求 3x3 单应（src->dst），H[2,2]=1。"""
    A, B = [], []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v]); B.append(v)
    h = np.linalg.solve(np.array(A, dtype=np.float64),
                        np.array(B, dtype=np.float64))
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def apply_perspective(img, labels, rng):
    """轻微 keystone（模拟斜拍）：四角向外 0-4% 扩张保证无空洞，标签随单应映射。"""
    w, h = img.size
    src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    cx, cy = w / 2.0, h / 2.0
    dst = []
    for x, y in src:
        fx = 1.0 + rng.uniform(0.0, 0.04)
        fy = 1.0 + rng.uniform(0.0, 0.04)
        dst.append((cx + (x - cx) * fx, cy + (y - cy) * fy))
    H = solve_homography(src, dst)
    Minv = np.linalg.inv(H)
    Minv /= Minv[2, 2]
    warped = img.transform((w, h), Image.PERSPECTIVE,
                           tuple(Minv.ravel()[:8]),
                           resample=Image.BICUBIC)

    def fwd(pt):
        x, y = pt
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d,
                (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d)

    out = []
    for cid, ncx, ncy, nbw, nbh in labels:
        x0, y0 = (ncx - nbw / 2) * w, (ncy - nbh / 2) * h
        x1, y1 = (ncx + nbw / 2) * w, (ncy + nbh / 2) * h
        pts = [fwd((x0, y0)), fwd((x1, y0)), fwd((x1, y1)), fwd((x0, y1))]
        uxs = [p[0] for p in pts]
        uys = [p[1] for p in pts]
        ua = (max(uxs) - min(uxs)) * (max(uys) - min(uys))
        xs = np.clip(uxs, 0, w)
        ys = np.clip(uys, 0, h)
        bx0, bx1 = float(xs.min()), float(xs.max())
        by0, by1 = float(ys.min()), float(ys.max())
        ncx2 = (bx0 + bx1) / 2 / w
        ncy2 = (by0 + by1) / 2 / h
        if not (0.0 <= ncx2 <= 1.0 and 0.0 <= ncy2 <= 1.0):
            continue
        if (bx1 - bx0) * (by1 - by0) < 0.55 * ua or bx1 - bx0 < 2 or by1 - by0 < 2:
            continue
        out.append((cid, ncx2, ncy2, (bx1 - bx0) / w, (by1 - by0) / h))
    return warped, out


def add_glare(img, rng):
    """镜面高光：1-2 条大范围柔和亮斑 + 一个小热点，模拟玻璃/亮面反光。"""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    for _ in range(rng.randint(1, 2)):
        cw, ch = int(rng.uniform(0.5, 1.1) * w), int(rng.uniform(0.2, 0.5) * h)
        streak = Image.new("L", (cw, ch), 0)
        sd = ImageDraw.Draw(streak)
        sd.ellipse([cw * 0.05, ch * 0.15, cw * 0.95, ch * 0.85], fill=255)
        streak = streak.rotate(rng.uniform(0, 180), expand=True)
        peak = rng.randint(45, 95)
        streak = streak.point(lambda a: int(a / 255 * peak))
        pos = (rng.randint(-cw // 2, w - cw // 2),
               rng.randint(-ch // 2, h - ch // 2))
        mask.paste(streak, pos, streak)
    d = ImageDraw.Draw(mask)
    hr = rng.uniform(0.03, 0.07) * w
    hx, hy = rng.uniform(0, w), rng.uniform(0, h)
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=rng.randint(90, 150))
    mask = mask.filter(ImageFilter.GaussianBlur(w * rng.uniform(0.015, 0.04)))
    white = Image.new("RGB", (w, h), (255, 253, 246))
    return Image.composite(white, img, mask)


# -------------------------------------------------------- post-process -----
def post_process(img, rng, nrng):
    arr = np.array(img, dtype=np.float32) / 255.0
    # gamma
    gamma = rng.uniform(0.8, 1.25)
    arr = np.power(np.clip(arr, 0, 1), gamma)
    # brightness & contrast
    arr *= rng.uniform(0.85, 1.15)
    arr = (arr - 0.5) * rng.uniform(0.85, 1.15) + 0.5
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    # small hue jitter (different tile-set color casts)
    dh = int(rng.uniform(-9, 9))
    if dh:
        hsv = np.array(img.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + dh) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    # gaussian noise
    sigma = rng.uniform(1.0, 4.0)
    arr = np.array(img, dtype=np.float32)
    arr += nrng.normal(0, sigma, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # occasional slight blur
    if rng.random() < 0.20:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.0)))
    return img


# ---------------------------------------------------------------- main -----
def save_sample(img, labels, img_path, lbl_path):
    img.save(img_path, quality=90)
    with open(lbl_path, "w") as f:
        for cid, cx, cy, bw, bh in labels:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def draw_preview(img, labels, out_path):
    img = img.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    for cid, cx, cy, bw, bh in labels:
        x0 = (cx - bw / 2) * CANVAS; y0 = (cy - bh / 2) * CANVAS
        x1 = (cx + bw / 2) * CANVAS; y1 = (cy + bh / 2) * CANVAS
        color = (255, 60, 60) if cid == BACK_ID else (255, 220, 40)
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        d.text((x0 + 2, max(0, y0 - 15)), CLASS_NAMES[cid], fill=color, font=font)
    img.save(out_path, quality=92)


def parse_back_range(s):
    lo, hi = s.split(",")
    return (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "training/datasets/mj-synth-v1"))
    ap.add_argument("--canvas", type=int, default=640,
                    help="合成图边长（640 或 1280）；牌面长边范围按比例缩放")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name-offset", type=int, default=0,
                    help="文件名编号偏移（多进程分片写同一目录时防重名）")
    ap.add_argument("--persp-prob", type=float, default=0.0,
                    help="每图施加轻透视（keystone）的概率")
    ap.add_argument("--glare-prob", type=float, default=0.0,
                    help="每图施加镜面高光的概率")
    ap.add_argument("--strong-shadow-prob", type=float, default=0.0,
                    help="每图使用强阴影（长深投影）的概率")
    ap.add_argument("--tiles-min", type=int, default=1)
    ap.add_argument("--tiles-max", type=int, default=15)
    ap.add_argument("--empty-prob", type=float, default=0.05)
    ap.add_argument("--back-prob", type=float, default=0.08,
                    help="per-tile back probability when no forced plan")
    ap.add_argument("--back-range", type=parse_back_range, default=None,
                    help="boost mode: per-image back fraction 'LO,HI' e.g. 0.3,0.5")
    ap.add_argument("--preview", type=int, default=0,
                    help="generate N annotated previews and exit")
    ap.add_argument("--preview-dir",
                     default=str(ROOT / "training/assets/preview"))
    ap.add_argument("--preview-prefix", default="preview_")
    args = ap.parse_args()

    global CANVAS, TILE_LONG_RANGE
    if args.canvas != 640:
        CANVAS = args.canvas
        TILE_LONG_RANGE = (60.0 * args.canvas / 640.0, 170.0 * args.canvas / 640.0)

    tiles = load_tiles()

    if args.preview > 0:
        out = Path(args.preview_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.preview):
            img, labels = generate_image(args.seed * 10_000 + 900_000 + i, tiles, args)
            draw_preview(img, labels, out / f"{args.preview_prefix}{i}.jpg")
            print(f"{args.preview_prefix}{i}.jpg: {len(labels)} tiles:",
                  [CLASS_NAMES[c] for c, *_ in labels])
        return

    out = Path(args.out)
    counts = np.zeros(NC, dtype=np.int64)
    for split, n in (("train", args.train), ("val", args.val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        empty = 0
        split_off = 0 if split == "train" else 500_000
        for i in range(n):
            seed = args.seed * 1_000_000 + split_off + i
            img, labels = generate_image(seed, tiles, args)
            name = f"mj_{split}_{i + args.name_offset:05d}"
            save_sample(img, labels,
                        out / "images" / split / f"{name}.jpg",
                        out / "labels" / split / f"{name}.txt")
            if not labels:
                empty += 1
            for cid, *_ in labels:
                counts[cid] += 1
            if (i + 1) % 250 == 0:
                print(f"[{split}] {i + 1}/{n}", flush=True)
        print(f"[{split}] done: {n} images, {empty} empty-label")

    yaml = out / "data.yaml"
    names = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(CLASS_NAMES))
    yaml.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnc: {NC}\nnames:\n{names}\n")

    print("\n=== class distribution ===")
    total = counts.sum()
    for i, nm in enumerate(CLASS_NAMES):
        print(f"{i:2d} {nm:6s} {counts[i]:6d}  {counts[i] / max(total, 1) * 100:5.2f}%")
    print(f"total instances: {total}")


if __name__ == "__main__":
    main()
