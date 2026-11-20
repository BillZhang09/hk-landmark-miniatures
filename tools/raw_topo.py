import os, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]              # the repository
sys.path.insert(0, str(ROOT))
from hk3d.fbx import load_textured
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
D = os.environ.get('HK3D_DATA', str(ROOT.parent))    # folder holding the LandsD sheet folders
B = {'boc': '11-SW-14A/BUILDING/B346831557701063A0', 'ifc2': '11-SW-8B/BUILDING/B344541627001063A0',
     'center': '11-SW-8D/BUILDING/B339821618801063A0', 'ckc': '11-SW-9C/BUILDING/B345831563001063A0',
     'hsbc': '11-SW-8D/BUILDING/B344751569101063A0', 'adm': '11-SW-14A/BUILDING/B349581546901063A0'}
for name, p in B.items():
    d = load_textured(f'{D}/{p}/{p.split("/")[-1]}.fbx')
    V, F = d['V'], d['F']
    # weld vertices that coincide (1 mm) before judging topology
    q = np.round(V * 1000).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True); inv = inv.ravel()
    G = inv[F]
    deg = (G[:, 0] == G[:, 1]) | (G[:, 1] == G[:, 2]) | (G[:, 0] == G[:, 2])
    G = G[~deg]
    E = np.sort(np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]]), axis=1)
    _, c = np.unique(E, axis=0, return_counts=True)
    n = inv.max() + 1
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    used = np.unique(G)
    ncomp = len(np.unique(lab[used]))
    P = V[F]; area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nz = np.abs(nrm[:, 2]) / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)
    ext = V.max(0) - V.min(0)
    print(f"{name}: verts {len(V)} tris {len(F)} | size {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} m, z {V[:,2].min():.1f}..{V[:,2].max():.1f} mPD"
          f" | degenerate {int(deg.sum())} | edges: open {int((c==1).sum())}, shared-by-2 {int((c==2).sum())}, by>2 {int((c>2).sum())}"
          f" | pieces {ncomp} | area {area.sum():.0f} m2, vertical walls {area[nz<0.1].sum()/area.sum()*100:.0f}%"
          f" | textured tris {(d['tex']>=0).mean()*100:.0f}%, textures {len(d['files'])}")
