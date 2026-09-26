"""Check the manuscript equations independently against the archived feature code."""
import ast
import json
import math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
source = root / 'centeredsquare_fusion_v1/common.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ('physical_descriptors', 'pressure_gauge')]
scope = dict(np=np, math=math, EPS=1e-12)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)

def equations(u, p, t, w):
    eps = 1e-12
    p = p - (p @ w / w.sum())[:, None]
    normu = lambda x: np.sqrt(np.sum(x*x*w[:, None]))
    normp = lambda x: np.sqrt(np.sum(x*x*w))
    U = np.array([normu(x)**2 for x in u])
    P = np.array([normp(x)**2 for x in p])
    h = np.maximum(np.diff(t), eps)
    v = np.diff(u, axis=0)/h[:, None, None]
    g = np.diff(np.log(np.maximum(U, eps)))/h
    su = np.sqrt(np.maximum(U, eps))
    sp = np.sqrt(np.maximum(P, eps))
    return np.array([.5*np.log(max(U[2]/w.sum(),eps)),
        .5*np.log(max(P[2]/w.sum(),eps)), np.log1p(normu(v[1])/(su[2]+eps)),
        np.log1p(normu(v[0])/(su[1]+eps)), np.tanh(g[1]), np.tanh(g[1]-g[0]),
        np.log1p(normp(p[2]-p[1])/(h[1]*(sp[2]+eps))),
        np.log1p(normu(v[1]-v[0])/(su[2]+eps))], dtype=np.float32)

rng = np.random.default_rng(20260917)
maxerr = 0.
for i in range(200):
    w = rng.uniform(.1,2,17)
    u,p = rng.normal(size=(3,17,2)),rng.normal(size=(3,17))
    t = np.cumsum(rng.uniform(.01,1,3))
    if i == 0: u *= 0; p *= 0
    a = scope['physical_descriptors'](u,p,t,w)
    b = equations(u,p,t,w)
    np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(a,scope['physical_descriptors'](u,p+12.5,t,w),rtol=2e-6,atol=2e-6)
    maxerr = max(maxerr,float(np.max(np.abs(a-b))))
result = dict(passed=True,cases=200,max_absolute_difference=maxerr,
              tests=['zero fields','irregular positive time intervals','pressure gauge shift'],
              scope='equations vs archived source; cached feature provenance checked separately')
(Path(__file__).parents[1]/'descriptor_equations_test.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
