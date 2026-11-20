"""Geometry-vs-texture check of LandsD building FBX: is facade detail real geometry?

For each tower: mesh stats, area-weighted facade relief (how much vertical-wall
area lies a small distance behind/in front of its facade plane), and an
orthographic render of the main facade three ways: texture, geometry shading,
and depth relative to the facade plane.
"""
import sys, numpy as np
from pathlib import Path
APP, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(APP))
from hk3d.fbx import load_textured
import open3d as o3d
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = {'中銀大廈 BoC': '11-SW-14A/BUILDING/B346831557701063A0', 'IFC2': '11-SW-8B/BUILDING/B344541627001063A0',
     '中環中心 The Center': '11-SW-8D/BUILDING/B339821618801063A0', '長江集團中心 CKC': '11-SW-9C/BUILDING/B345831563001063A0',
     '滙豐總行 HSBC': '11-SW-8D/BUILDING/B344751569101063A0', '金鐘辦公樓 Admiralty': '11-SW-14A/BUILDING/B349581546901063A0'}
if len(sys.argv) > 4:  # optional "name=sheet/BUILDING/ID" pairs replace the six-tower set
    B = dict(a.split('=', 1) for a in sys.argv[4:])
from hk3d.fbx import load_meshes
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']
imgs = {}

def tex_img(f):
    if f not in imgs:
        im = Image.open(f).convert('RGB'); imgs[f] = np.asarray(im)
    return imgs[f]

rows = []
for name, p in B.items():
    fp = str(DATA / p / (p.split('/')[-1] + '.fbx'))
    d = load_textured(fp)
    if d is None:  # FBX0: geometry only
        ms = load_meshes(fp); off_ = np.cumsum([0] + [len(m[1]) for m in ms])
        d = {'V': np.concatenate([m[1] for m in ms]),
             'F': np.concatenate([m[2] + o for m, o in zip(ms, off_)]), 'tex': None}
    V, F = d['V'], d['F']
    P = V[F]; n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); a = 0.5 * np.linalg.norm(n, axis=1)
    nn = n / np.maximum(2 * a[:, None], 1e-12)
    vert = np.abs(nn[:, 2]) < 0.1
    # edge lengths
    el = np.linalg.norm(np.concatenate([P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]]), axis=1)
    # dominant facade direction by vertical area (angle mod 180)
    ang = np.degrees(np.arctan2(nn[vert, 1], nn[vert, 0])) % 180
    h, e = np.histogram(ang, bins=180, range=(0, 180), weights=a[vert])
    h = h + np.roll(h, 1) + np.roll(h, -1)
    th = np.radians(e[np.argmax(h)] + 0.5)
    dirs = []
    for sgn in (1, -1):
        dvec = sgn * np.array([np.cos(th), np.sin(th), 0])
        sel = vert & (nn @ dvec > 0.985)
        c = P[sel].mean(1)
        off = c @ dvec
        if sel.sum() == 0: continue
        main = np.average(off, weights=a[sel]) if False else None
        # outermost plane with >=5% of this direction's area
        hh, ee = np.histogram(off, bins=np.arange(off.min() - .05, off.max() + .1, .05), weights=a[sel])
        cum = np.cumsum(hh[::-1]) / hh.sum()
        dirs.append((sgn, dvec, sel, off, a[sel]))
    # relief: per direction, area of facing faces within (0.05, 1.5] m behind the plane holding the most area
    rel = []
    for sgn, dvec, sel, off, aa in dirs:
        hh, ee = np.histogram(off, bins=np.arange(off.min() - .025, off.max() + .05, .05), weights=aa)
        top = ee[np.argmax(hh)] + .025
        dz = off - top
        small = (np.abs(dz) > 0.075) & (np.abs(dz) <= 1.5)
        rel.append(aa[small].sum() / aa.sum())
    rows.append((name, len(V), len(F), a.sum(), a[vert].sum() / a.sum(), np.median(el), (el < 0.5).mean(),
                 np.mean(rel), V[:, 2].max()))

    # ---- render main facade (direction with larger area)
    sgn, dvec, sel, off, aa = max(dirs, key=lambda t: t[4].sum())
    u = np.array([-dvec[1], dvec[0], 0])
    cu = V @ u; z = V[:, 2]
    res = max(0.15, (z.max() - z.min()) / 1400)
    us = np.arange(cu.min(), cu.max(), res); zs = np.arange(z.max(), z.min(), -res)
    U, Z = np.meshgrid(us, zs)
    far = (V @ dvec).max() + 50
    org = U[..., None] * u + Z[..., None] * np.array([0, 0, 1]) + far * dvec
    ray = np.concatenate([org, np.broadcast_to(-dvec, org.shape)], -1).reshape(-1, 6).astype(np.float32)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(V.astype(np.float32), F.astype(np.uint32))
    r = sc.cast_rays(o3d.core.Tensor(ray))
    t = r['t_hit'].numpy(); pid = r['primitive_ids'].numpy().astype(np.int64); buv = r['primitive_uvs'].numpy()
    hit = np.isfinite(t); pid = np.where(hit, pid, 0)
    H, W = U.shape
    # texture
    col = np.full((len(t), 3), 255, np.uint8)
    idx = np.where(hit)[0]
    for k in (np.unique(d['tex'][pid[idx]]) if d['tex'] is not None else []):
        if k < 0: continue
        img = tex_img(d['files'][k]); ih, iw = img.shape[:2]
        m = idx[d['tex'][pid[idx]] == k]
        w = np.column_stack([1 - buv[m].sum(1), buv[m]])
        uv = (d['uv'][pid[m]] * w[:, :, None]).sum(1)
        x = np.clip((uv[:, 0] % 1) * iw, 0, iw - 1).astype(int); y = np.clip((1 - uv[:, 1] % 1) * ih, 0, ih - 1).astype(int)
        col[m] = img[y, x]
    col = col.reshape(H, W, 3)
    # geometry shading
    nrm = nn[pid.clip(0)] ; nrm[~hit] = 0
    nrm *= np.sign((nrm @ dvec))[:, None] + (nrm @ dvec == 0)[:, None]
    light = np.array([0.45, 0.0, 0.0]) + 0.0
    L = 0.55 * dvec + 0.6 * u + 0.58 * np.array([0, 0, 1]); L /= np.linalg.norm(L)
    sh = np.clip(nrm @ L, 0, 1) * 0.85 + 0.15; sh[~hit] = 1
    sh = sh.reshape(H, W)
    # depth relative to the facade plane (per row median so setbacks don't dominate)
    depth = np.where(hit, t, np.nan).reshape(H, W)
    ref = np.nanmedian(depth, axis=1, keepdims=True)
    rd = depth - ref
    fig, ax = plt.subplots(1, 3, figsize=(15, 15 * H / W / 3 + 0.8), dpi=110)
    ax[0].imshow(col); ax[0].set_title('照片貼圖' if d['tex'] is not None else '（無紋理 FBX0，沒有貼圖）')
    ax[1].imshow(sh, cmap='gray', vmin=0, vmax=1); ax[1].set_title('純幾何（無貼圖）')
    im = ax[2].imshow(rd, cmap='RdBu', vmin=-1.0, vmax=1.0); ax[2].set_title('相對立面深度 ±1 m')
    for x in ax: x.set_xticks([]); x.set_yticks([])
    fig.colorbar(im, ax=ax[2], fraction=0.04, label='m（紅＝凸出）')
    fig.suptitle(f'{name}：主立面正投影，每像素實地 {res:.2f} m')
    fig.tight_layout(); fig.savefig(OUT / f"{p.split('/')[-1]}.png"); plt.close(fig)
    # fraction of facade pixels whose depth differs from row median by 0.1..1.5 m
    fin = np.isfinite(rd)
    print(f"{name}: facade pixels with 0.1-1.5 m relief {((np.abs(rd)>0.1)&(np.abs(rd)<=1.5))[fin].mean()*100:.1f}%", flush=True)

print('\nname | verts | tris | area m2 | vertical% | median edge m | edges<0.5m | facade area 0.075-1.5 m off main plane | top mPD')
for r_ in rows:
    print(f"{r_[0]} | {r_[1]} | {r_[2]} | {r_[3]:.0f} | {r_[4]*100:.0f}% | {r_[5]:.2f} | {r_[6]*100:.0f}% | {r_[7]*100:.1f}% | {r_[8]:.1f}")
