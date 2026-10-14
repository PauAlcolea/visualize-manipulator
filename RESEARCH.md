# Task-based link-length optimization

## Useful primary sources

1. **Thais Campos and Hadas Kress-Gazit (2021), “Synthesizing Modular Manipulators
   For Tasks With Time, Obstacle, And Torque Constraints.”**
   [Paper and full text](https://arxiv.org/html/2106.09487v1).
   Particularly relevant: Sections 2.2, 3.3, 3.5–3.7. The authors jointly search
   robot structure and configurations under task constraints, then verify the
   result. Their broader system includes paths, obstacles, and dynamics. It
   highlights why reaching isolated points does not guarantee a traversable path.
   Their formulation does not constrain end-effector orientation; this app does.

2. **Kento Kawaharazuka, Kei Okada, and Masayuki Inaba (IROS 2024), “Robot Design
   Optimization with Rotational and Prismatic Joints using Black-Box
   Multi-Objective Optimization.”**
   [Paper](https://arxiv.org/abs/2409.20038),
   [full text](https://arxiv.org/html/2409.20038v1).
   Directly relevant to compact robot design: it optimizes joint count and link
   lengths for a task and explores Pareto alternatives. It is useful for a future
   extension that also selects joint types/topology. The present implementation
   fixes topology and minimizes one scalar length objective instead.

3. **SciPy SLSQP documentation.**
   [Official reference](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
   Documents the constrained sequential quadratic programming implementation,
   stopping criteria, and iteration controls used here. It references Kraft's
   1988 SQP software report and Nocedal & Wright's *Numerical Optimization*.

These sources motivate the problem formulation; the app is a simplified,
independent implementation, not a reproduction of either paper's algorithm.

## Implemented formulation

For selected links `S`, choose shared lengths `L_i` and one joint vector `q_k`
for each of up to three requested poses `(p_k, R_k)`:

```text
minimize     sum(L_i, i in S)
subject to   min_i <= L_i <= max_i
             lower_j <= q_kj <= upper_j          for every pose and joint
             ||FK_position(L, q_k) - p_k|| <= 2 mm
             ||Log(R_k.T @ FK_rotation(L,q_k))|| <= 2 degrees
             z of every chain endpoint >= 0    when the table is enabled
```

Each selected nonzero offset retains its direction and is scaled to `L_i`.
The sum of all link lengths differs from the selected-link objective only by
a constant, since unselected offsets and tool length are fixed. Default variables
are the shoulder and forearm offsets; base height is fixed unless selected.
Zero-offset wrist axes cannot acquire a link direction through this optimizer.

The base is at the world origin, on an infinite table. Links are straight,
zero-thickness segments. Each segment's height interpolates linearly between
its endpoints, so endpoint nonnegativity checks the entire segment. Contact is
allowed; numerical penetration tolerance is 1e-6 mm. Manual posing can still
show an invalid configuration, labeled TABLE COLLISION; it is excluded from
accepted solver results and sampled workspace coverage.

## Numerical method and verification

`design.py` uses deterministic multistart search. Each start initializes shared
lengths, obtains per-target IK seeds, and runs bounded least squares to approach
feasibility. SLSQP then minimizes total selected length with nonlinear pose and
table constraints. Lengths are scaled by 500 mm and joint variables use radians.
Locked joints are removed from the optimization variables.

Internal pose constraints use 1.99 mm and 1.99° to leave a small verification
margin below the advertised 2 mm / 2°. A shorter design can deliberately consume
that tolerance; it is not an exact-pose solution. Every returned candidate is
checked again using forward kinematics, actual rotation distance, bounds, and
table clearance. Optimizer success alone is never enough. The shortest verified
candidate across the starts is retained, even if minimization did not converge;
that status is explicitly reported. No candidate means “not found,” not a proof
that the specification is impossible.

## Practical limits

- This is a nonconvex local search, not a global optimality certificate.
- Reaching three poses does not establish workspace coverage or a collision-free
  path between them. Motion planning would be a separate next step.
- Link thickness, joint housings, self-collision, stiffness, torque, payload, and
  manipulability are absent. A mathematically shortest arm may be a poor physical
  design or sit near singularities. Minimum length bounds let you impose packaging
  requirements, but do not replace a mechanical model.
- A later design objective could trade length against manipulability and torque,
  followed by path and collision verification, as motivated by the papers above.
