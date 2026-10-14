# Reach — robot workspace explorer

A local Python app for exploring serial robotic arm geometry before mechanical design.
Starts with six revolute joints; supports 1–10 joints, local axes, link offsets,
joint limits, a tool extension, and full position/orientation inverse kinematics.

## Run

Use the existing environment in the parent directory:

```sh
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python app.py
```

Open http://127.0.0.1:8050. The app binds only to localhost. No CDN or external
service is required at runtime. Stop with Ctrl+C.

## Explore

1. Edit link offsets, axes, and limits in the geometry table, then **Apply geometry & limits**.
   Use **+ Joint**, row deletion, or **Reset 6 DOF** to change the chain.
2. **Save baseline**, edit a length or joint limit, and Apply. The old geometry is
   overlaid in amber, using the same sample count, orientation filter, and cell size.
3. Enter target XYZ in mm and roll/pitch/yaw in degrees, then **Find a pose**.
   The arm moves to the solution, or to the best attempt if none was found.
   Uncheck **Match full orientation** for a position-only query.
4. Move the joint sliders beside the model to explore poses live. The solver uses
   that pose as its first starting point. Editing a target preserves the manual pose
   and clears the previous solve message. **Use current arm pose as target** captures
   a known reachable XYZ/orientation. Live position and orientation errors appear
   under the model. If an orientation solve fails, a position-only diagnostic checks
   whether XYZ can be reached with the same limits.
5. Switch among filled cells, point cloud, and arm-only views. Cross-sections show
   occupied cells in XY, XZ, or YZ. A slice coordinate selects its containing cell slab.
6. Enable the orientation filter to retain samples within the chosen full-rotation
   angular tolerance of the target. Increase sample count for better coverage.
7. Export the applied geometry as JSON. Import loads the editor; press Apply to use it.
   Baselines are session-local, so export designs you want to keep.
8. Enable **Table at z = 0** to enforce an infinite table for workspace samples,
   IK, and design optimization. Manual poses below the table show a collision warning.
9. In **Design for three target poses**, enter all three XYZ/RPY targets, choose
   which nonzero link offsets can change, and set minimum/maximum lengths. The
   default optimizes shoulder and forearm, leaving base height and tool fixed.
   **Find shortest design** reports the shortest verified candidate across 4 or 12
   starts. **Apply verified design** saves the old geometry as a baseline. Load a
   target slot into the main target panel and solve to inspect the new arm.

See [RESEARCH.md](RESEARCH.md) for papers, the optimization formulation, numerical
method, and limitations. All three full orientations are enforced, independently
of the single-target panel's position-only checkbox. The target pose tolerance is
2 mm / 2°, not exact equality; minimizing length may use most of that tolerance.

## Geometry convention

World +Z is up. Each row applies a rotation about its **local** X/Y/Z axis, then a
translation by its ΔX/ΔY/ΔZ in the resulting frame. The next row starts there.
The tool extends along the final local +X axis. Zero offsets give intersecting
wrist axes. Equal min/max values lock a joint. All joints are revolute.

Target Euler angles use fixed/world XYZ: `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
The solid tool frame and dashed target frame use X red, Y green, Z blue.

## What the results mean

- Forward kinematics and deterministic scrambled Sobol sampling generate workspace
  samples within the applied joint limits.
- Filled cells are the union of occupied voxels, with exposed faces rendered. They
  preserve gaps instead of wrapping everything in a convex hull, but they do **not**
  guarantee every point inside an occupied cell is reachable. Empty cells may have
  been missed by sampling. Cell size affects appearance, not solver precision.
- The orientation filter uses the shortest 3D rotation angle, not independent Euler
  angle differences. It is an approximate sampled map, not an exact dexterous workspace.
  Tight filters can miss almost all samples even when the target has an IK solution.
- Target IK uses bounded, multistart nonlinear least squares with 16 deterministic
  starting poses (including the displayed pose). A solution must meet 2 mm position
  and 2° full-orientation tolerances. Failure is inconclusive. It is not an exact
  reachability proof or a trajectory planner. The optional table constraint checks
  every straight link centerline; contact at z = 0 is allowed.
- With the table disabled, points below ground are retained. General collision
  geometry, self-collision, link thickness, torque, loads, and dynamics are not modeled.
- Large geometries may use a coarser effective cell size to bound rendering memory;
  the actual size is displayed. Point rendering is thinned at higher sample counts.

## Validate

```sh
../.venv/bin/python -m unittest discover -s tests -v
```

Implementation: NumPy kinematics, SciPy rotation math/Sobol sampling/least squares,
and a Plotly Dash interface. See the official [Dash callback documentation](https://dash.plotly.com/basic-callbacks)
and [SciPy Euler-angle convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html).
