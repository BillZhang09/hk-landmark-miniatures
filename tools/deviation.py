"""How far is our printed model from the raw LandsD mesh (real-world metres)?

    deviation.py <prefix> <name...>      the jobs cache/jobs/<prefix>_<name>/
    deviation.py <model.stl...>          a print folder: <stem>_result.json
                                         (or result.json) beside each model

The model's centre, scale and base height come from the build's
result.json, and the raw building is the one the build carved (its
``landmark.files``); a build without that record falls back to the
buildings listed in ``B`` below."""
import os, sys, json, numpy as np, trimesh, open3d as o3d
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]              # the repository
sys.path.insert(0, str(ROOT))
from hk3d.fbx import load_textured
D = os.environ.get('HK3D_DATA', str(ROOT.parent)); J = f"{os.environ.get('HK3D_CACHE', ROOT / 'cache')}/jobs"
# fallback building files for builds that did not record theirs
B = {'boc': '11-SW-14A/BUILDING/B346831557701063A0', 'ifc2': '11-SW-8B/BUILDING/B344541627001063A0',
     'icc': '11-NW-24C/BUILDING/B345521825901063A0',
     'center': '11-SW-8D/BUILDING/B339821618801063A0', 'ckc': '11-SW-9C/BUILDING/B345831563001063A0',
     'hsbc': '11-SW-8D/BUILDING/B344751569101063A0', 'adm': '11-SW-14A/BUILDING/B349581546901063A0'}


def locate(arg, prefix):
    """(name, model stl, result.json) for a job name or a model path."""
    if arg.endswith('.stl'):
        stl = Path(arg)
        for res in (stl.with_name(stl.stem + '_result.json'), stl.with_name('result.json')):
            if res.exists():
                return stl.stem, stl, res
        sys.exit(f"{stl}: no result.json beside it")
    job = Path(f'{J}/{prefix}_{arg}')
    return arg, job / f'{arg}.stl', job / 'result.json'


def raw_mesh(name, r):
    files = r.get('stats', {}).get('landmark', {}).get('files') or []
    if not files:
        if name.lower() not in B:
            sys.exit(f"{name}: the build recorded no landmark files and none is listed in B")
        p = B[name.lower()]
        files = [f'{p}/{p.split("/")[-1]}.fbx']
    Vs, Fs, k = [], [], 0
    for f in files:
        d = load_textured(f'{D}/{f}')
        if d is None:
            sys.exit(f"{f}: could not read the raw mesh")
        Vs.append(d['V'].astype(np.float64)); Fs.append(np.asarray(d['F']) + k); k += len(d['V'])
    return np.concatenate(Vs), np.concatenate(Fs)


T = lambda a: o3d.core.Tensor(np.ascontiguousarray(a, np.float32))
def scene(V, F):
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(T(V), o3d.core.Tensor(np.ascontiguousarray(F, np.uint32))); return sc
def sample(V, F, n, rng):
    P = V[F]; cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); a = 0.5 * np.linalg.norm(cr, axis=1)
    i = rng.choice(len(F), n, p=a / a.sum()); r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    pts = P[i, 0] * (1 - r1)[:, None] + P[i, 1] * (r1 * (1 - r2))[:, None] + P[i, 2] * (r1 * r2)[:, None]
    return pts, cr[i] / np.maximum(np.linalg.norm(cr[i], axis=1, keepdims=True), 1e-12)
def stat(d):
    return f"median {np.median(d):.2f} m, 90% {np.percentile(d,90):.2f}, 99% {np.percentile(d,99):.2f}, max {d.max():.1f} | within 0.25 m {np.mean(d<=0.25)*100:.0f}%, 0.5 m {np.mean(d<=0.5)*100:.0f}%, 1 m {np.mean(d<=1)*100:.0f}%"
args = sys.argv[1:]
prefix = None if args[0].endswith('.stl') else args.pop(0)
for arg in args:
    name, stl, res = locate(arg, prefix); rng = np.random.default_rng(0)
    r = json.load(open(res)); P = r['params']
    V, F = raw_mesh(name, r)
    V = V - np.array([P['e'], P['n'], 0.0])                          # local metres, like the model
    s = P['diameter_mm'] / (2 * P['radius_m'])                       # mm per metre
    m = trimesh.load(stl, process=False)
    W = np.asarray(m.vertices, np.float64); G = np.asarray(m.faces)
    zs_ = s * P.get('z_scale', 1.0)                                  # print mm per real metre, vertically
    W = np.column_stack([W[:, 0] / s, W[:, 1] / s, (W[:, 2] - P['base_mm']) / zs_])   # metres above the model's floor
    raw, ours = scene(V, F), scene(W, G)
    # raw surface seen from outside: a ray leaving along +-normal (or tilted) escapes the raw mesh
    pts, nr = sample(V, F, 300000, rng)
    ext = np.zeros(len(pts), bool)
    t1 = np.cross(nr, [0, 0, 1.0]); t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9); t2 = np.cross(nr, t1)
    for sg in (1, -1):
        for tilt in (None, t1, -t1, t2, -t2):
            dr = sg * nr if tilt is None else (sg * nr + 0.8 * tilt); dr = dr / np.linalg.norm(dr, axis=1, keepdims=True)
            rays = np.concatenate([pts + dr * 0.02, dr], 1)
            ext |= ~np.isfinite(raw.cast_rays(T(rays))['t_hit'].numpy())
    # vertical datum: the elevation at the top of the base, as the build
    # recorded it.  A datum fitted to roofs and ledges is printed alongside:
    # if the two differ, the whole model sits at the wrong height, which a
    # fitted datum alone would hide.
    floor = P['floor_m'] if P.get('floor_m') is not None else r['stats']['elev_min_m']
    flat = ext & (nr[:, 2] > 0.9)                 # roofs and ledges facing up, not soffits
    sub = pts[flat][::max(1, flat.sum() // 20000)]
    fit = floor
    for span, step in ((4.0, 0.1), (0.15, 0.01)):
        zs = np.arange(fit - span, fit + span + 1e-9, step)
        cost = [np.mean(np.minimum(ours.compute_distance(T(sub - [0, 0, z])).numpy(), 1.0)) for z in zs]
        fit = float(zs[int(np.argmin(cost))])
    d1 = ours.compute_distance(T(pts[ext] - [0, 0, floor])).numpy()
    # our model is watertight, so "inside" is well defined: a far raw point
    # inside it was buried (a recess or an open shell filled in), one outside
    # it is geometry we lost
    occ = ours.compute_occupancy(T(pts[ext] - [0, 0, floor])).numpy() > 0.5
    zrel = (pts[ext][:, 2] - V[:, 2].min()) / (V[:, 2].max() - V[:, 2].min())
    q, _ = sample(W, G, 600000, rng); q = q + [0, 0, floor]
    lo, hi = V.min(0), V.max(0)
    # our surface over the building's own footprint (what the raw mesh
    # covers seen from above, plus one 0.5 m cell), from 2 m above its lowest
    # point: podiums, undercrofts and crowns, not the ground and neighbours
    g = 0.5
    gx, gy = np.arange(lo[0] - 2, hi[0] + 2, g), np.arange(lo[1] - 2, hi[1] + 2, g)
    GX, GY = np.meshgrid(gx, gy)
    down = np.stack([GX, GY, np.full_like(GX, hi[2] + 50), 0 * GX, 0 * GX, -np.ones_like(GX)], -1).reshape(-1, 6)
    from scipy import ndimage
    fp = ndimage.binary_dilation(np.isfinite(raw.cast_rays(T(down))['t_hit'].numpy()).reshape(GX.shape))
    qi, qj = np.floor((q[:, 1] - gy[0]) / g).astype(int), np.floor((q[:, 0] - gx[0]) / g).astype(int)
    ok = (qi >= 0) & (qi < fp.shape[0]) & (qj >= 0) & (qj < fp.shape[1])
    keep = ok & (q[:, 2] > lo[2] + 2.0)
    keep[ok] &= fp[qi[ok], qj[ok]]
    d2 = raw.compute_distance(T(q[keep])).numpy()
    z2 = (q[keep][:, 2] - lo[2]) / (hi[2] - lo[2])
    print(f"== {name}  (1:{round(1000/s)}, base top {floor:.2f} mPD (fit to roofs {fit:.2f}), raw exterior share {ext.mean()*100:.0f}%)")
    print("  raw -> ours, all   :", stat(d1))
    print("  raw -> ours, top10%:", stat(d1[zrel > 0.9]))
    print("  raw -> ours, rest  :", stat(d1[zrel <= 0.9]))
    # the lowest part stands in the ground and podium of the full model, so
    # raw points there are inside it by construction
    base = zrel < 0.12
    print("  raw -> ours, above the base (lowest 12%):", stat(d1[~base]))
    for lim in (0.5, 1.0):
        far = d1 > lim
        print(f"  raw points > {lim} m away: {far.mean()*100:.1f}% = buried in our solid {np.mean(far & occ)*100:.1f}% (base {np.mean(far & occ & base)*100:.1f}%) + lost {np.mean(far & ~occ)*100:.1f}%"
              f"  (top 10%: buried {np.mean((far & occ)[zrel > 0.9])*100:.1f}% + lost {np.mean((far & ~occ)[zrel > 0.9])*100:.1f}%)")
    print("  ours -> raw, footprint:", stat(d2))
    print("  ours -> raw, top10%:", stat(d2[z2 > 0.9]))
    np.savez(f'{sys.argv[0][:-3]}_{prefix or stl.parent.name}_{name}.npz', pts=pts[ext], nrm=nr[ext], d1=d1, occ=occ, q=q[keep], d2=d2, floor=floor)
