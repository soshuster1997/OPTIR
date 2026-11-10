"""
PTIR exporter (Google Colab compatible)

Exports from a Photothermal .ptir (HDF5) file:
  - spectra        -> one CSV per spectrum + one combined CSV (with stage X/Y position rows)
                      + spectra_positions.csv
  - optical images -> TIFF with scale bar, TIFF without (pixel-size metadata),
                      and TIFF with scale bar + spectrum locations marked/labelled
  - "Heightmaps"   -> CSV + 32-bit float TIFF (with pixel-size metadata), plus an annotated 8-bit RGB
                      rendering (colormap + scale bar + spectrum locations) in heightmaps/annotated/

Coordinate convention (verified on the example file by registering the IR maps against the
camera image): an image's PositionX/Y is its CENTER in stage coordinates (um); stage +X is
image right, stage +Y is image UP (so image row index increases as stage Y decreases).
Map arrays are stored in display orientation (no flip needed).

Colab usage (paste into cells):

    !pip -q install h5py tifffile
    from google.colab import files
    up = files.upload()                       # upload this file AND your .ptir
    import ptir_export
    out = ptir_export.export_ptir(next(k for k in up if k.endswith(".ptir")))
    ptir_export.zip_and_download(out)
"""
import math
import os
import re
import shutil

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _s(v):
    """HDF5 attribute -> clean python str / float."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0].item()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _attr(ds, key, default=None):
    return _s(ds.attrs[key]) if key in ds.attrs else default


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")


def _numbered(group):
    """Children of an HDF5 group sorted by their trailing number."""
    return sorted(group.keys(), key=lambda k: int(re.findall(r"\d+$", k)[0]) if re.findall(r"\d+$", k) else -1)


def _pixel_size_um(ds, arr):
    """um per pixel, computed from the stored array shape (not the 'Pixels' attribute,
    because stored optical images can be down-sampled relative to it)."""
    w_um, h_um = _attr(ds, "SizeWidth"), _attr(ds, "SizeHeight")
    if not w_um or not h_um:
        return None
    return float(w_um) / arr.shape[1], float(h_um) / arr.shape[0]


def _nice_bar_um(width_um, target_frac=0.2):
    target = width_um * target_frac
    cands = [m * 10 ** e for e in range(-1, 5) for m in (1, 2, 5)]
    return min(cands, key=lambda c: abs(c - target) if c <= width_um * 0.35 else 1e9)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_scale_bar(rgb, px_um):
    """Returns (image array, bar length in um, bounding box of bar+label)."""
    img = Image.fromarray(rgb).convert("RGB")
    W, H = img.size
    bar_um = _nice_bar_um(W * px_um)
    bar_px = int(round(bar_um / px_um))
    thick = max(4, int(H * 0.008))
    margin = int(W * 0.04)
    x1, y1 = W - margin, H - margin
    x0, y0 = x1 - bar_px, y1 - thick
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(0, 0, 0))
    d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    fsize = max(12, int(H * 0.032))
    d.text(((x0 + x1) / 2, y0 - 8), f"{bar_um:g} \u00b5m", fill=(255, 255, 255), font=_font(fsize),
           anchor="mb", stroke_width=2, stroke_fill=(0, 0, 0))
    bbox = (x0 - 10, y0 - fsize - 20, x1 + 10, y1 + 10)
    return np.asarray(img), bar_um, bbox


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _mark_spectra(rgb, px_um, center_um, spectra, avoid_box=None, color=(255, 220, 0)):
    """Draw a ring+crosshair and a label (name + stage X, Y in um) for each spectrum inside the image."""
    img = Image.fromarray(rgb).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    font = _font(max(11, int(H * 0.022)))
    r = max(8, int(H * 0.012))
    placed = [avoid_box] if avoid_box else []
    pts = []
    for s in spectra:
        px = W / 2 + (s["x"] - center_um[0]) / px_um[0]
        py = H / 2 - (s["y"] - center_um[1]) / px_um[1]          # stage +Y is image up
        if 0 <= px < W and 0 <= py < H:
            pts.append((s, px, py))
    # labels must not cover any marker either
    for _, px, py in pts:
        placed.append((px - r * 2, py - r * 2, px + r * 2, py + r * 2))
    n = 0
    for s, px, py in pts:
        # marker: black halo then colored ring + crosshair with a gap in the middle
        d.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2], outline=(0, 0, 0), width=4)
        d.ellipse([px - r, py - r, px + r, py + r], outline=color, width=2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = r * 0.35, r * 1.8
            d.line([px + dx * a, py + dy * a, px + dx * b, py + dy * b], fill=(0, 0, 0), width=4)
            d.line([px + dx * a, py + dy * a, px + dx * b, py + dy * b], fill=color, width=2)
        text = f"{s['label']}\n({s['x']:.1f}, {s['y']:.1f}) \u00b5m"
        l, t, rr, bb = d.multiline_textbbox((0, 0), text, font=font, spacing=2, stroke_width=2)
        tw, th = rr - l, bb - t
        gap = r * 2
        pos, leader = None, False
        # try spots around the marker at growing distances (right side first); far spots get a leader line
        for rad in (gap, gap * 2, gap * 3.5, gap * 5, gap * 7, gap * 10, gap * 14):
            for ang in (0, 30, 330, 60, 300, 90, 270, 120, 240, 150, 210, 180):
                ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                ax, ay = px + rad * ca, py - rad * sa
                x0 = ax if ca > 0.3 else (ax - tw if ca < -0.3 else ax - tw / 2)
                y0 = ay - th if sa > 0.3 else (ay if sa < -0.3 else ay - th / 2)
                box = (x0, y0, x0 + tw, y0 + th)
                if box[0] >= 0 and box[1] >= 0 and box[2] <= W and box[3] <= H and not any(_overlap(box, q) for q in placed):
                    pos, leader = box, rad > gap
                    break
            if pos:
                break
        if pos is None:  # crowded: clamp the first candidate into the image
            x0, y0 = min(max(px + gap, 0), W - tw), min(max(py - th / 2, 0), H - th)
            pos, leader = (x0, y0, x0 + tw, y0 + th), True
        placed.append(pos)
        if leader:
            qx, qy = min(max(px, pos[0]), pos[2]), min(max(py, pos[1]), pos[3])
            d.line([px, py, qx, qy], fill=(0, 0, 0), width=4)
            d.line([px, py, qx, qy], fill=color, width=1)
        d.multiline_text((pos[0] - l, pos[1] - t), text, fill=color, font=font, spacing=2,
                         stroke_width=2, stroke_fill=(0, 0, 0))
        n += 1
    return np.asarray(img), n


def _render_map(arr, px_um, cmap="viridis", pct=(1, 99), target=1000):
    """8-bit RGB display rendering of a float map (percentile stretch + colormap, integer
    nearest-neighbour upscaling so small maps are legible). Returns (rgb, px_um of rendering)."""
    from matplotlib import colormaps
    lo, hi = np.nanpercentile(arr, pct)
    norm = np.clip((arr - lo) / (hi - lo if hi > lo else 1), 0, 1)
    rgb = (colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    k = max(1, int(round(target / max(arr.shape))))
    if k > 1:
        rgb = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    return rgb, (px_um[0] / k, px_um[1] / k)


def _wavenumber_axis(m, n_points):
    """Wavenumber axis of a measurement. Normally stored in Spectroscopic_Values, but some spectra
    (e.g. ones added after the fact in PTIR Studio) don't have that dataset, so fall back to
    RangeWavenumberStart + Increment * index, which matches the stored axis exactly when both exist."""
    if "Spectroscopic_Values" in m:
        x = np.asarray(m["Spectroscopic_Values"][()], dtype=float).reshape(-1)
        if len(x) == n_points:
            return np.round(x, 4)
    start, inc = _attr(m, "RangeWavenumberStart"), _attr(m, "RangeWavenumberIncrement")
    if start is None or inc is None:
        raise ValueError(f"{m.name}: no Spectroscopic_Values and no RangeWavenumber* attributes")
    return np.round(float(start) + float(inc) * np.arange(n_points), 4)


def _res_kwargs(px_um):
    """TIFF resolution tags in pixels/cm (standard unit; Fiji/ImageJ reads this as a calibrated scale)."""
    if px_um is None:
        return {}
    return dict(resolution=(1e4 / px_um[0], 1e4 / px_um[1]), resolutionunit="CENTIMETER")


# ----------------------------------------------------------------------------
# main export
# ----------------------------------------------------------------------------
def export_ptir(ptir_path, out_dir=None, include_background=False, verbose=True):
    base = os.path.splitext(os.path.basename(ptir_path))[0]
    out_dir = out_dir or f"{base}_export"
    dirs = {k: os.path.join(out_dir, k) for k in ("spectra", "optical_images", "heightmaps")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(dirs["heightmaps"], "annotated"), exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    with h5py.File(ptir_path, "r") as f:
        # ------------------------------------------------------------ spectra
        combined = None
        col_pos = {}          # combined-CSV column -> (stage X, stage Y)
        spec_info = []        # for the position table + image markers
        n_spec = 0
        for mname in [k for k in _numbered(f) if k.startswith("Measurement_")]:
            m = f[mname]
            if not include_background and _attr(m, "IsBackground", 0):
                log(f"  skip {mname} ({_attr(m, 'Label')}, background)")
                continue
            label = _attr(m, "Label", mname)
            sx, sy = (round(v, 2) if v is not None else None for v in (_attr(m, "LocationX"), _attr(m, "LocationY")))
            for cname in [k for k in _numbered(m) if k.startswith("Channel_")]:
                ch = m[cname]
                if "Raw_Data" not in ch:
                    continue
                y = np.asarray(ch["Raw_Data"][()]).reshape(-1)
                x = _wavenumber_axis(m, len(y))
                ch_label = _attr(ch, "Label", cname)
                unit = (re.findall(r"\(([^)]*)\)", ch_label) or ["a.u."])[-1]
                wn_unit = _attr(m, "RangeWavenumberUnits", "cm\u207b\u00b9")
                col = f"{label}/{ch_label} [{unit}]"
                df = pd.DataFrame({f"X value [{wn_unit}]": x, col: y})
                df.to_csv(os.path.join(dirs["spectra"], f"{mname}_{_safe(label)}.csv"), index=False)
                n_spec += 1
                xcol = df.columns[0]
                combined = df if combined is None else combined.merge(df, on=xcol, how="outer")
                col_pos[col] = (sx, sy)
                log(f"  spectrum {mname} '{col}' ({len(x)} pts) at stage ({sx}, {sy}) um")
            if sx is not None and sy is not None:
                spec_info.append(dict(measurement=mname, label=label, x=float(sx), y=float(sy),
                                      z=round(_attr(m, "LocationZ", float("nan")), 2)))

        if combined is not None:
            xcol = combined.columns[0]
            combined = combined.sort_values(xcol)
            # same convention as PTIR Studio: drop rows with no data in any spectrum
            combined = combined.dropna(how="all", subset=combined.columns[1:])
            # position rows go on top of the data (first column holds the row name)
            pos_rows = pd.DataFrame(
                [{xcol: "Stage X position [\u00b5m]", **{c: p[0] for c, p in col_pos.items()}},
                 {xcol: "Stage Y position [\u00b5m]", **{c: p[1] for c, p in col_pos.items()}}],
                columns=combined.columns)
            out = pd.concat([pos_rows, combined.astype(object)], ignore_index=True)
            out.to_csv(os.path.join(out_dir, f"{base}_all_spectra.csv"), index=False)
        if spec_info:
            pd.DataFrame(spec_info).rename(columns={"x": "stage_x_um", "y": "stage_y_um", "z": "stage_z_um"}) \
                .to_csv(os.path.join(out_dir, "spectra_positions.csv"), index=False)

        # ------------------------------------------------------ optical images
        if "Images" in f:
            for name in _numbered(f["Images"]):
                ds = f["Images"][name]
                arr = ds[()]
                label = _attr(ds, "Label", name)
                rgb = arr[..., :3] if arr.ndim == 3 else np.stack([arr] * 3, -1)  # alpha dropped
                px = _pixel_size_um(ds, rgb)
                stem = f"{name}_{_safe(label)}"
                odir = dirs["optical_images"]
                tifffile.imwrite(os.path.join(odir, stem + "_noscale.tif"), rgb, photometric="rgb", **_res_kwargs(px))
                if not px:
                    log(f"  image {name} '{label}': NO size info in file -> no scale bar")
                    continue
                barred, bar_um, bar_box = _draw_scale_bar(rgb, px[0])
                tifffile.imwrite(os.path.join(odir, stem + "_scalebar.tif"), barred, photometric="rgb", **_res_kwargs(px))
                msg = f"  image {name} '{label}': {px[0]:.4f} um/px, {bar_um:g} um bar"
                cxy = (_attr(ds, "PositionX"), _attr(ds, "PositionY"))
                if None not in cxy and spec_info:
                    marked, n = _mark_spectra(barred, px, cxy, spec_info, avoid_box=bar_box)
                    if n:
                        tifffile.imwrite(os.path.join(odir, stem + "_scalebar_spectra.tif"), marked,
                                         photometric="rgb", **_res_kwargs(px))
                    msg += f", {n} spectra marked"
                log(msg)

        # ---------------------------------------------------------- heightmaps
        if "Heightmaps" in f:
            for name in _numbered(f["Heightmaps"]):
                ds = f["Heightmaps"][name]
                arr = np.asarray(ds[()], dtype=np.float32)
                label = _attr(ds, "Label", name)
                stem = f"{name}_{_safe(label)}"
                np.savetxt(os.path.join(dirs["heightmaps"], stem + ".csv"), arr, delimiter=",", fmt="%.7g")
                tifffile.imwrite(os.path.join(dirs["heightmaps"], stem + ".tif"), arr,
                                 photometric="minisblack", **_res_kwargs(_pixel_size_um(ds, arr)))
                px = _pixel_size_um(ds, arr)
                if px:
                    # display rendering (8-bit RGB, 1-99 percentile colour stretch) with scale bar + spectra marks;
                    # the float TIFF/CSV above remain the real data
                    rgb, rpx = _render_map(arr, px)
                    rgb, _, bar_box = _draw_scale_bar(rgb, rpx[0])
                    cxy = (_attr(ds, "PositionX"), _attr(ds, "PositionY"))
                    n = 0
                    if None not in cxy and spec_info:
                        rgb, n = _mark_spectra(rgb, rpx, cxy, spec_info, avoid_box=bar_box, color=(255, 255, 255))
                    tifffile.imwrite(os.path.join(dirs["heightmaps"], "annotated", stem + "_annotated.tif"),
                                     rgb, photometric="rgb", **_res_kwargs(rpx))
                    log(f"    -> annotated rendering, {n} spectra marked")
                log(f"  map {name} '{label}' {arr.shape} units={_attr(ds, 'UnitPrefix', '')}{_attr(ds, 'Units', '')}"
                    f" channel={_attr(ds, 'Channel')}")

    log(f"\nDone: {n_spec} spectra. Output in ./{out_dir}")
    return out_dir


def zip_and_download(out_dir):
    zp = shutil.make_archive(out_dir, "zip", out_dir)
    try:
        from google.colab import files
        files.download(zp)
    except ImportError:
        print("Not in Colab; zip is at", zp)
    return zp


if __name__ == "__main__":
    import sys
    export_ptir(sys.argv[1])
