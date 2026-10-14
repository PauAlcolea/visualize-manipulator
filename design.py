"""Task-based dimensional synthesis with a shared geometry and per-task joints.

Multistart feasibility search followed by constrained length minimization.
See RESEARCH.md for the formulation, sources, and limitations.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from kinematics import forward, orientation, angular_error, solve


def length_rows(model):
    rows = []
    for i, j in enumerate(model['joints']):
        length = float(np.linalg.norm([j[k] for k in ('x', 'y', 'z')]))
        if length > 1e-8:
            rows.append(dict(index=i, name=f'J{i+1} · {j["name"]}', current=round(length, 3),
                             optimize='yes' if i in (1, 2) else 'no', minimum=20,
                             maximum=max(800, round(length * 2))))
    return rows


def parse_targets(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise ValueError('Enter one to three target poses.')
    targets = []
    for row in rows:
        try:
            values = np.array([float(row[k]) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
        except (KeyError, ValueError, TypeError):
            raise ValueError('Every target needs numeric XYZ and roll/pitch/yaw.') from None
        if not np.isfinite(values).all():
            raise ValueError('Target values must be finite.')
        targets.append((values[:3], orientation(*values[3:])))
    return targets


def optimize_lengths(model, targets, settings, restarts=4):
    targets = parse_targets(targets)
    if model.get('table') and any(p[2] < -1e-6 for p, _ in targets):
        raise ValueError('A target is below the table (z < 0). Move it or disable the table.')
    selected, directions, original, length_bounds = [], [], [], []
    for row in settings or []:
        if row.get('optimize') != 'yes':
            continue
        try:
            i = int(row['index'])
            if i < 0 or i >= len(model['joints']) or i in selected:
                raise ValueError()
            v = np.array([model['joints'][i][k] for k in ('x', 'y', 'z')], dtype=float)
            length = float(np.linalg.norm(v))
            lo, hi = float(row['minimum']), float(row['maximum'])
            if length < 1e-8 or not np.isfinite([lo, hi]).all() or not 1 <= lo < hi <= 5000:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ValueError('Selected links need nonzero offsets and bounds 1 ≤ min < max ≤ 5000 mm.') from None
        selected.append(i)
        directions.append(v / length)
        original.append(length)
        length_bounds.append((lo / 500, hi / 500))
    if not selected:
        raise ValueError('Select at least one nonzero link to optimize.')
    n, m, count = len(model['joints']), len(targets), len(selected)
    qlo = np.deg2rad([j['lower'] for j in model['joints']])
    qhi = np.deg2rad([j['upper'] for j in model['joints']])
    active = qhi - qlo > 1e-9
    na = int(active.sum())
    bounds = length_bounds + list(zip(qlo[active], qhi[active])) * m
    lower, upper = np.array(bounds, dtype=float).T

    def unpack(x):
        candidate = deepcopy(model)
        for i, direction, length in zip(selected, directions, x[:count] * 500):
            for k, value in zip(('x', 'y', 'z'), direction * length):
                candidate['joints'][i][k] = float(value)
        q = np.tile(qlo, (m, 1))
        q[:, active] = x[count:].reshape(m, na)
        return candidate, np.rad2deg(q)

    def errors(x):
        candidate, q = unpack(x)
        p, r, chain = forward(candidate, q)
        dp = p - np.array([t[0] for t in targets])
        dr = Rotation.from_matrix(np.array([t[1].T for t in targets]) @ r).as_rotvec()
        return dp, dr, chain

    def residual(x):
        dp, dr, chain = errors(x)
        result = np.r_[dp.ravel() / 100, dr.ravel()]
        if model.get('table'):
            result = np.r_[result, np.minimum(chain[:, 1:, 2], 0).ravel() / 10]
        return result

    def constraints(x):
        dp, dr, chain = errors(x)
        # Slightly tighter internal tolerances leave a numerical verification margin.
        result = np.r_[1 - np.sum((dp / 1.99)**2, axis=1),
                       1 - np.sum((dr / np.deg2rad(1.99))**2, axis=1)]
        if model.get('table'):
            result = np.r_[result, chain[:, 1:, 2].ravel() / 500]
        return result

    best = None

    def consider(x, converged):
        nonlocal best
        if not np.isfinite(x).all() or np.any(x < lower - 1e-8) or np.any(x > upper + 1e-8):
            return
        candidate, q = unpack(x)
        checks = []
        for angles, (target, rotation) in zip(q, targets):
            p, r, chain = forward(candidate, angles)
            distance = float(np.linalg.norm(p - target))
            angle = float(angular_error(r, rotation))
            clearance = float(np.min(chain[:, :, 2])) if chain.ndim == 3 else float(np.min(chain[:, 2]))
            if distance > 2 or angle > 2 or (model.get('table') and clearance < -1e-6):
                return
            checks.append(dict(angles=angles.tolist(), distance=distance, angle=angle, clearance=clearance))
        total = float(np.sum(x[:count]) * 500)
        if best is None or total < best['length']:
            best = dict(model=candidate, checks=checks, length=total, original_length=float(sum(original)),
                        lengths=[dict(index=i, before=before, after=float(after * 500))
                                 for i, before, after in zip(selected, original, x[:count])],
                        converged=bool(converged))

    rng = np.random.default_rng(23)
    for attempt in range(restarts):
        lengths = np.clip(np.array(original) / 500, lower[:count], upper[:count]) if attempt == 0 else rng.uniform(lower[:count], upper[:count])
        x = np.r_[lengths, np.tile((qlo[active] + qhi[active]) / 2, m)]
        starting_model, _ = unpack(x)
        for k, (target, rotation) in enumerate(targets):
            ik = solve(starting_model, target, rotation, restarts=4)
            x[count + k * na:count + (k+1) * na] = np.deg2rad(ik['angles'])[active]
        fit = least_squares(residual, np.clip(x, lower, upper), bounds=(lower, upper),
                            max_nfev=220, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        consider(fit.x, False)
        opt = minimize(lambda z: float(np.sum(z[:count])), fit.x, method='SLSQP',
                       bounds=bounds, constraints=[dict(type='ineq', fun=constraints)],
                       options=dict(maxiter=220, ftol=1e-9))
        consider(opt.x, opt.success)
    if best is None:
        return dict(found=False, message='No verified design found within these bounds. This is not proof of infeasibility; widen bounds or try different geometry.')
    return dict(found=True, **best)
