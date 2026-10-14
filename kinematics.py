"""Serial revolute chains. Distances: mm; public joint angles: degrees."""
from functools import lru_cache
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


def default_joints():
    return [
        dict(name='Base', axis='Z', x=0, y=0, z=120, lower=-180, upper=180),
        dict(name='Shoulder', axis='Y', x=280, y=0, z=0, lower=-100, upper=100),
        dict(name='Elbow', axis='Y', x=240, y=0, z=0, lower=-135, upper=135),
        dict(name='Wrist roll', axis='X', x=0, y=0, z=0, lower=-180, upper=180),
        dict(name='Wrist pitch', axis='Y', x=0, y=0, z=0, lower=-100, upper=100),
        dict(name='Wrist yaw', axis='Z', x=0, y=0, z=0, lower=-180, upper=180),
    ]


def validate(rows, tool):
    if not isinstance(rows, list) or not rows or len(rows) > 10:
        raise ValueError('Use between 1 and 10 joints.')
    result = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Joint {i + 1} must be an object.')
        joint = dict(row)
        joint['name'] = str(row.get('name') or f'Joint {i + 1}')[:60]
        joint['axis'] = str(row.get('axis', '')).upper()
        if joint['axis'] not in ('X', 'Y', 'Z'):
            raise ValueError(f'Joint {i + 1}: axis must be X, Y, or Z.')
        for key in ('x', 'y', 'z', 'lower', 'upper'):
            try:
                joint[key] = float(row[key])
            except (ValueError, TypeError, KeyError):
                raise ValueError(f'Joint {i + 1}: {key} must be a number.') from None
            if not np.isfinite(joint[key]):
                raise ValueError('All values must be finite.')
        if joint['lower'] > joint['upper']:
            raise ValueError(f'Joint {i + 1}: minimum must not exceed maximum.')
        if max(abs(joint[k]) for k in ('x', 'y', 'z')) > 5000:
            raise ValueError('Keep each link offset within ±5000 mm.')
        if max(abs(joint[k]) for k in ('lower', 'upper')) > 360:
            raise ValueError('Keep joint limits within ±360°.')
        result.append(joint)
    if tool is None or not np.isfinite(float(tool)) or not 0 <= float(tool) <= 5000:
        raise ValueError('Tool length must be between 0 and 5000 mm.')
    return dict(joints=result, tool=float(tool))


def forward(model, angles):
    """Rotate about local joint axis, then translate in that rotated frame.

    Accepts (joints,) or (samples, joints); returns tip, rotation, and chain points.
    Tool extends along the final frame's +X axis.
    """
    q = np.asarray(angles, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    rotation = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    position = np.zeros((len(q), 3))
    points = [position.copy()]
    for i, joint in enumerate(model['joints']):
        v = np.zeros((len(q), 3))
        v[:, 'XYZ'.index(joint['axis'])] = np.deg2rad(q[:, i])
        rotation = rotation @ Rotation.from_rotvec(v).as_matrix()
        offset = np.array([joint[k] for k in ('x', 'y', 'z')])
        position = position + np.einsum('nij,j->ni', rotation, offset)
        points.append(position.copy())
    position = position + rotation[:, :, 0] * model['tool']
    points.append(position.copy())
    chain = np.stack(points, axis=1)
    return (position[0], rotation[0], chain[0]) if single else (position, rotation, chain)


def orientation(roll, pitch, yaw):
    """Fixed/world XYZ Euler angles: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()


def angular_error(rotations, target):
    relative = np.swapaxes(rotations, -1, -2) @ target
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


@lru_cache(maxsize=8)
def sampled_workspace(model_json, power):
    model = json.loads(model_json)
    joints = model['joints']
    lower = np.array([j['lower'] for j in joints], dtype=float)
    upper = np.array([j['upper'] for j in joints], dtype=float)
    unit = qmc.Sobol(len(joints), scramble=True, seed=42).random_base2(power)
    positions, rotations, chain = forward(model, lower + unit * (upper - lower))
    if model.get('table', False):
        valid = np.min(chain[:, :, 2], axis=1) >= -1e-6
        positions, rotations = positions[valid], rotations[valid]
    return positions, rotations


def workspace(model, power=14, target_rotation=None, tolerance=15):
    positions, rotations = sampled_workspace(json.dumps(model, sort_keys=True), power)
    if target_rotation is not None:
        positions = positions[angular_error(rotations, target_rotation) <= tolerance]
    return positions


def solve(model, target_position, target_rotation, initial=None, restarts=16,
          position_tolerance=2.0, angle_tolerance=2.0):
    """Bounded multistart IK. Failure is inconclusive, not an unreachability proof."""
    joints = model['joints']
    # Browser JSON serializes whole-number limits as integers. The working angle
    # vector must still be floating point or expand() truncates finite-difference
    # steps, giving the optimizer a zero Jacobian and freezing it at its seed.
    lower = np.array([j['lower'] for j in joints], dtype=float)
    upper = np.array([j['upper'] for j in joints], dtype=float)
    active = upper - lower > 1e-9
    target_position = np.asarray(target_position)
    length = max(sum(np.linalg.norm([j[k] for k in ('x', 'y', 'z')]) for j in joints)
                 + model['tool'], 1)

    def expand(x):
        q = lower.copy()
        q[active] = x
        return q

    def residual(x):
        p, r, chain = forward(model, expand(x))
        error = (p - target_position) / length
        if target_rotation is not None:
            error = np.r_[error, Rotation.from_matrix(target_rotation.T @ r).as_rotvec()]
        if model.get('table', False):
            error = np.r_[error, 20 * np.minimum(chain[1:, 2], 0) / length]
        return error

    rng = np.random.default_rng(17)
    seeds = [np.clip(initial, lower, upper) if initial is not None else (lower + upper) / 2]
    seeds += [lower + rng.random(len(joints)) * (upper - lower) for _ in range(restarts - 1)]
    best = None
    for seed in seeds:
        if active.any():
            fit = least_squares(residual, seed[active], bounds=(lower[active], upper[active]),
                                max_nfev=180, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = expand(fit.x)
        else:
            q = lower.copy()
        p, r, chain = forward(model, q)
        distance = float(np.linalg.norm(p - target_position))
        angle = float(angular_error(r, target_rotation)) if target_rotation is not None else 0.0
        score = max(distance / position_tolerance, angle / angle_tolerance)
        clearance = float(np.min(chain[:, 2]))
        if model.get('table', False):
            score = max(score, max(0, -clearance) / 1e-6)
        candidate = dict(angles=q.tolist(), distance=distance, angle=angle, found=score <= 1,
                         clearance=clearance)
        if best is None or score < best[0]:
            best = score, candidate
        if candidate['found'] or not active.any():
            break
    return best[1]
