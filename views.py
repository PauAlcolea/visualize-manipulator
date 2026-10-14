"""Plotly views; occupied cells preserve gaps without a convex hull."""
import numpy as np
import plotly.graph_objects as go
from kinematics import forward

TEAL = '#24c5b1'
AMBER = '#f3b35b'


def cells(points, size):
    return np.unique(np.floor(points / size).astype(int), axis=0)


def cell_mesh(points, size, color, name, opacity):
    occupied = set(map(tuple, cells(points, size)))
    vertices, triangles = [], []
    # Emit only exposed faces, retaining internal cavities and disconnected cells.
    for cell in occupied:
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                if tuple(neighbor) in occupied:
                    continue
                start = len(vertices)
                for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = np.array(cell, dtype=float)
                    corner[axis] += sign == 1
                    corner[other] += [u, v]
                    vertices.append(corner * size)
                triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    if not vertices:
        return go.Mesh3d(name=name)
    xyz = np.array(vertices).T
    ijk = np.array(triangles).T
    return go.Mesh3d(x=xyz[0], y=xyz[1], z=xyz[2], i=ijk[0], j=ijk[1], k=ijk[2],
                     color=color, opacity=opacity, name=name, showlegend=True,
                     flatshading=True, hoverinfo='skip')


def frame(fig, position, rotation, length, name, dash=None):
    for axis, color in enumerate(('#f07d7d', '#8dd69d', '#7ea5ff')):
        end = position + rotation[:, axis] * length
        xyz = np.stack([position, end]).T
        fig.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode='lines',
                                  line=dict(color=color, width=5, dash=dash or 'solid'),
                                  name=f'{name} {"XYZ"[axis]}', showlegend=False,
                                  hovertemplate=f'{name} +{"XYZ"[axis]}<extra></extra>'))


def scene(model, angles, points, baseline, target_position, target_rotation, mode, size):
    fig = go.Figure()
    for p, color, name, opacity in ((baseline, AMBER, 'Saved baseline', .12),
                                     (points, TEAL, 'Current workspace', .24)):
        if p is None or len(p) == 0 or mode == 'hidden':
            continue
        if mode == 'cells':
            fig.add_trace(cell_mesh(p, size, color, name, opacity))
        else:
            # Deterministic display thinning; computation and slice still use all samples.
            p = p[::max(1, len(p) // 18000)]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                                      marker=dict(size=2, color=color, opacity=.22),
                                      name=name, hoverinfo='skip'))
    tip, rotation, chain = forward(model, angles)
    fig.add_trace(go.Scatter3d(x=chain[:, 0], y=chain[:, 1], z=chain[:, 2],
                              mode='lines+markers', line=dict(color='#e7edf5', width=12),
                              marker=dict(size=5, color='#e7edf5', line=dict(color='#10202d', width=2)),
                              name='Arm', hovertemplate='(%{x:.0f}, %{y:.0f}, %{z:.0f}) mm<extra>Arm</extra>'))
    fig.add_trace(go.Scatter3d(x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
                              mode='markers', marker=dict(size=7, color=AMBER, symbol='diamond'),
                              name='Target'))
    span = max(sum(np.linalg.norm([j[k] for k in ('x', 'y', 'z')]) for j in model['joints'])
               + model['tool'], 100)
    if model.get('table'):
        fig.add_trace(go.Mesh3d(x=[-span, span, span, -span], y=[-span, -span, span, span],
                                z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                                color='#6c879e', opacity=.25, name='Table · z = 0', showlegend=True,
                                hoverinfo='name'))
    frame(fig, tip, rotation, span * .11, 'Tool')
    frame(fig, np.array(target_position), target_rotation, span * .11, 'Target', 'dash')
    fig.update_layout(template='plotly_dark', paper_bgcolor='#101c2a', plot_bgcolor='#101c2a',
                      margin=dict(l=0, r=0, b=0, t=0), height=590, uirevision='arm-view',
                      legend=dict(orientation='h', x=.02, y=1),
                      scene=dict(aspectmode='cube', bgcolor='#101c2a', uirevision='workspace-camera',
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=.9)),
                                 **{f'{a}axis': dict(title=f'{a.upper()} · mm', range=[-span, span],
                                    backgroundcolor='#101c2a', gridcolor='#263342', zerolinecolor='#536271')
                                    for a in 'xyz'}))
    return fig


def slice_view(points, baseline, plane, level, size, target):
    axes = {'XY': (0, 1, 2), 'XZ': (0, 2, 1), 'YZ': (1, 2, 0)}[plane]
    a, b, normal = axes
    selected = []
    for p in (points, baseline):
        if p is None:
            selected.append(np.empty((0, 2), dtype=int))
        else:
            c = cells(p, size)
            selected.append(c[c[:, normal] == int(np.floor(level / size))][:, [a, b]])
    all_cells = np.concatenate(selected)
    fig = go.Figure()
    if len(all_cells):
        lo, hi = all_cells.min(axis=0), all_cells.max(axis=0)
        # Cell size is bounded by UI and geometry extent, so these arrays stay small.
        z = np.full(tuple(hi - lo + 1), np.nan)
        for c in selected[1]:
            z[tuple(c - lo)] = 1
        for c in selected[0]:
            index = tuple(c - lo)
            z[index] = 3 if z[index] == 1 else 2
        fig.add_trace(go.Heatmap(x=(np.arange(lo[0], hi[0] + 1) + .5) * size,
                                y=(np.arange(lo[1], hi[1] + 1) + .5) * size, z=z.T,
                                zmin=1, zmax=3, showscale=False,
                                colorscale=[[0, AMBER], [.249, AMBER], [.25, TEAL],
                                            [.749, TEAL], [.75, '#b3d7b4'], [1, '#b3d7b4']],
                                hovertemplate='%{x:.0f}, %{y:.0f} mm<extra>Sampled cell</extra>'))
    else:
        fig.add_annotation(text='No samples in this slice', x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
    if int(np.floor(target[normal] / size)) == int(np.floor(level / size)):
        fig.add_trace(go.Scatter(x=[target[a]], y=[target[b]], mode='markers',
                                marker=dict(symbol='x', color='white', size=12), showlegend=False))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#101c2a', plot_bgcolor='#101c2a',
                      height=300, margin=dict(l=55, r=20, t=10, b=45),
                      xaxis=dict(title=f'{"XYZ"[a]} · mm', constrain='domain'),
                      yaxis=dict(title=f'{"XYZ"[b]} · mm', scaleanchor='x', scaleratio=1))
    return fig
