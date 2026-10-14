"""Local robot workspace explorer. Run with ../.venv/bin/python app.py."""
import json
import numpy as np
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, ctx, no_update, Patch
from dash.exceptions import MissingCallbackContextException
from scipy.spatial.transform import Rotation
from kinematics import default_joints, validate, orientation, workspace, solve, forward, angular_error
from views import scene, slice_view, cells
from design import length_rows, optimize_lengths

app = Dash(__name__, title='Reach · Robot workspace', suppress_callback_exceptions=True)
DEFAULT = validate(default_joints(), 80)


def number(id, value, step=1, debounce=True, **kwargs):
    return dcc.Input(id=id, type='number', value=value, step=step, debounce=debounce, **kwargs)


def field(label, control):
    return html.Label([html.Span(label), control], className='field')


def card(title, subtitle, children, **kwargs):
    return html.Section([html.H2(title), html.P(subtitle, className='muted'), *children], className='card', **kwargs)


app.layout = html.Div([
    dcc.Store(id='model', data=DEFAULT), dcc.Store(id='baseline'), dcc.Store(id='solution'),
    dcc.Store(id='scene-traces'),
    dcc.Store(id='workspace-camera'),
    dcc.Store(id='design-result'),
    dcc.Download(id='download'),
    html.Header([
        html.Div([html.Div('R /', className='logo'), html.Div([html.H1('Reach'), html.P('ROBOT WORKSPACE EXPLORER')])], className='brand'),
        html.Div([html.Span('●', className='live-dot'), ' LOCAL KINEMATICS LAB'], className='tag'),
    ]),
    html.Div([html.Div([html.H2('Find the shape of possible.'),
                        html.P('Build a chain. Set its limits. Explore where your tool can go.')]),
              html.Div(id='model-badge', className='badge')], className='intro'),
    html.Div([
        html.Aside([
            card('01 / Arm geometry', 'Offsets are in each joint’s rotated local frame. All lengths are in mm.', [
                dash_table.DataTable(id='joints', data=default_joints(),
                    columns=[dict(name=label, id=key, type='text' if key in ('name', 'axis') else 'numeric',
                                  **({'presentation': 'dropdown'} if key == 'axis' else {}))
                             for key, label in [('name', 'Joint'), ('axis', 'Axis'), ('x', 'ΔX'), ('y', 'ΔY'),
                                                ('z', 'ΔZ'), ('lower', 'Min °'), ('upper', 'Max °')]],
                    editable=True, row_deletable=True,
                    dropdown={'axis': {'options': [dict(label=a, value=a) for a in 'XYZ']}},
                    style_table={'overflowX': 'auto'},
                    style_cell={'backgroundColor': '#101c2a', 'color': '#dce6ef', 'border': '1px solid #2a3847',
                                'fontFamily': 'inherit', 'fontSize': 12, 'padding': '9px 5px', 'minWidth': 44},
                    style_header={'backgroundColor': '#1b2939', 'fontWeight': 'bold'},
                    style_cell_conditional=[{'if': {'column_id': 'name'}, 'minWidth': 80, 'textAlign': 'left'}]),
                html.Div([html.Button('+ Joint', id='add'), html.Button('Reset 6 DOF', id='reset')], className='button-row'),
                field('Tool extension along local +X · mm', number('tool', 80, min=0, max=5000)),
                dcc.Checklist(id='table-enabled', options=[dict(label=' Table at z = 0 (applies immediately)', value='yes')], value=[]),
                html.P('Infinite table; straight, zero-thickness links may touch the surface but cannot pass below it. The base is mounted at z = 0.', className='note'),
                html.Button('Apply geometry & limits', id='apply', className='primary full'),
                html.Div(id='pending-changes', className='note warning'),
                html.Div(id='config-message', className='note'),
                html.Details([html.Summary('How to define a joint'), html.P('Each row rotates about its local X, Y, or Z axis, then moves by ΔX, ΔY, ΔZ to the next joint. Zero offsets allow intersecting wrist axes. Equal limits lock a joint. Delete a row with ×. Edits take effect when you press Apply.')]),
                html.Div([html.Button('Save baseline', id='save-baseline'), html.Button('Clear', id='clear-baseline'),
                          html.Button('Export JSON', id='export')], className='button-row'),
                html.Div(id='baseline-message', className='note'),
                dcc.Upload(id='upload', children=html.Div('Import a saved geometry .json'), multiple=False, className='upload'),
            ]),
            card('02 / Target pose', 'Specify a tool-tip position and its full orientation.', [
                html.Div([field(a + ' · mm', number('target-' + a.lower(), v, 10, debounce=False)) for a, v in zip('XYZ', [400, 100, 250])], className='three-cols'),
                html.Div([field(a + ' · °', number('target-' + a.lower(), 0, 5, debounce=False)) for a in ['Roll', 'Pitch', 'Yaw']], className='three-cols'),
                html.P('Fixed XYZ angles: Rz(yaw) · Ry(pitch) · Rx(roll). Tool points along +X. Frame colors: X red, Y green, Z blue.', className='note'),
                dcc.Checklist(id='match-orientation', options=[{'label': ' Match full orientation', 'value': 'yes'}], value=['yes']),
                html.P('With this checked, reaching XYZ alone is not enough: the tool must also match roll, pitch, and yaw. Try position-only to separate the two.', className='note'),
                html.Button('Use current arm pose as target', id='capture-pose', className='full'),
                html.Button('Find a pose', id='solve', className='primary full'),
                dcc.Loading(html.Div(id='solve-message', children='Ready to solve. Tolerances: 2 mm and 2°.', className='solve-status'), type='dot', color='#24c5b1'),
            ]),
        ]),
        html.Main([
            html.Section([
                html.Div([html.Div([html.H2('Workspace'), html.P('Rotate to inspect · scroll to zoom', className='muted')]),
                          dcc.Dropdown(id='view-mode', options=[dict(label=l, value=v) for l, v in [('Filled cells', 'cells'), ('Point cloud', 'points'), ('Arm only', 'hidden')]],
                                       value='cells', clearable=False, className='view-select')], className='view-header'),
                html.Div([
                    dcc.Graph(id='scene', config={'displaylogo': False}, responsive=True),
                    html.Div([html.H2('Joint angles'),
                              html.P('Drag to move the arm live.', className='muted'),
                              html.Div(id='pose-controls')], className='pose-panel'),
                ], className='pose-workbench'),
                html.Div(id='live-pose', className='live-pose'),
                html.Div(id='metrics', className='metrics'),
            ], className='card scene-card'),
            card('Workspace sampling', 'The workspace updates as you apply geometry or change these controls.', [
                html.Div([
                    field('Samples', dcc.Dropdown(id='quality', options=[dict(label=l, value=v) for l, v in [('4,096 · fast', 12), ('16,384 · standard', 14), ('65,536 · detailed', 16)]], value=14, clearable=False)),
                    field('Cell size · mm', number('cell-size', 40, 10, min=10, max=500)),
                    field('Orientation tolerance · °', number('orientation-tolerance', 20, 5, min=1, max=180)),
                ], className='three-cols'),
                dcc.Checklist(id='filter-orientation', options=[dict(label=' Show only samples matching the target orientation', value='yes')], value=[]),
                html.P(id='sampling-note', className='note'),
            ]),
            card('Workspace cross-section', 'Teal: current · amber: baseline only · pale green: overlap. White ×: target within slice.', [
                html.Div([field('Plane', dcc.Dropdown(id='plane', options=['XY', 'XZ', 'YZ'], value='XY', clearable=False)),
                          field('Slice coordinate · mm', number('slice-level', 240, 20))], className='two-cols'),
                dcc.Graph(id='slice', config={'displaylogo': False}),
            ]),
            card('Design for three target poses', 'Minimize selected link lengths while reaching all three full poses within 2 mm and 2°. Joint limits and the table setting are enforced.', [
                dash_table.DataTable(id='design-targets',
                    data=[dict(name=f'Pose {i+1}', x=x, y=y, z=z, roll=0, pitch=0, yaw=0)
                          for i, (x,y,z) in enumerate([(400,100,250), (300,-200,200), (450,0,150)])],
                    columns=[dict(name=k.upper() if k in 'xyz' else k.title(), id=k,
                                  type='text' if k == 'name' else 'numeric')
                             for k in ['name','x','y','z','roll','pitch','yaw']], editable=True,
                    style_table={'overflowX':'auto'},
                    style_cell={'backgroundColor':'#101c2a','color':'#dce6ef','border':'1px solid #344557', 'padding':'9px','fontSize':12},
                    style_header={'backgroundColor':'#1b2939','fontWeight':'bold'}),
                html.Div([
                    dcc.Dropdown(id='design-slot', options=[dict(label=f'Pose {i+1}',value=i) for i in range(3)], value=0, clearable=False, style={'width':130}),
                    html.Button('Copy current target into slot', id='save-design-target'),
                    html.Button('Load slot into target', id='load-design-target'),
                ], className='button-row'),
                html.P('Lengths scale each selected offset along its existing direction. Zero-offset wrists, unselected links, joint axes, and tool extension stay fixed. Edit bounds in mm.', className='note'),
                dash_table.DataTable(id='design-lengths', data=length_rows(DEFAULT),
                    columns=[dict(name=label,id=k,type='text' if k in ('name','optimize') else 'numeric',
                                  editable=k in ('optimize','minimum','maximum'),
                                  **({'presentation':'dropdown'} if k=='optimize' else {}))
                             for k,label in [('name','Link'),('current','Current mm'),('optimize','Optimize?'),('minimum','Min mm'),('maximum','Max mm')]],
                    dropdown={'optimize':{'options':[dict(label='Yes',value='yes'),dict(label='No',value='no')]}},
                    style_table={'overflowX':'auto'},
                    style_cell={'backgroundColor':'#101c2a','color':'#dce6ef','border':'1px solid #344557','padding':'9px','fontSize':12}),
                field('Search effort', dcc.Dropdown(id='design-effort', options=[dict(label='4 starts · quick',value=4),dict(label='12 starts · thorough',value=12)], value=4, clearable=False)),
                html.Button('Find shortest design', id='optimize-design', className='primary full'),
                dcc.Loading(html.Div(id='design-message', className='solve-status'), type='dot', color='#24c5b1'),
                html.Div(id='design-stale', className='note'),
                html.Button('Apply verified design', id='apply-design', disabled=True, className='full'),
                html.P('Best verified candidate from a local multistart search; no global minimum guarantee. Checks static poses, not paths between them. Very short designs can be near singularities.', className='note'),
                html.Details([html.Summary('Method and research'),
                    html.P('Shared link lengths and one joint configuration per target are optimized together with constrained SLSQP. Every returned pose is checked independently with forward kinematics.'),
                    html.Ul([
                        html.Li(html.A('Campos & Kress-Gazit (2021): task-constrained robot synthesis',href='https://arxiv.org/abs/2106.09487',target='_blank')),
                        html.Li(html.A('Kawaharazuka et al. (2024): minimizing robot joint count and link lengths',href='https://arxiv.org/abs/2409.20038',target='_blank')),
                        html.Li(html.A('SciPy: constrained SLSQP optimization',href='https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html',target='_blank')),
                    ])]),
            ]),
            html.P('Geometric reach only: optional table checking uses link centerlines. Self-collision, link thickness, loads, and motor torque are not modeled. Filled cells contain sampled poses; the entire cell is not guaranteed reachable.', className='footnote'),
        ]),
    ], className='layout'),
])


@app.callback(Output('joints', 'data'), Output('tool', 'value'),
              Input('add', 'n_clicks'), Input('reset', 'n_clicks'),
              State('joints', 'data'), State('tool', 'value'), prevent_initial_call=True)
def edit_rows(add, reset, rows, tool):
    if ctx.triggered_id == 'reset':
        return default_joints(), 80
    if len(rows) < 10:
        rows = rows + [dict(name=f'Joint {len(rows) + 1}', axis='Z', x=100, y=0, z=0, lower=-90, upper=90)]
    return rows, tool


@app.callback(Output('model', 'data'), Output('config-message', 'children'),
              Input('apply', 'n_clicks'), State('joints', 'data'), State('tool', 'value'), State('table-enabled','value'), prevent_initial_call=True)
def apply_model(n, rows, tool, table):
    try:
        model = validate(rows, tool)
        model['table'] = bool(table)
        return model, 'Applied. Workspace and solver use these limits.'
    except ValueError as error:
        return no_update, str(error)


@app.callback(Output('model','data',allow_duplicate=True), Input('table-enabled','value'), State('model','data'), prevent_initial_call=True)
def toggle_table(enabled, model):
    return dict(model, table=bool(enabled))


@app.callback(Output('pending-changes', 'children'), Input('joints', 'data'),
              Input('tool', 'value'), Input('model', 'data'))
def pending_changes(rows, tool, model):
    try:
        edited = validate(rows, tool)
    except ValueError:
        return 'Editor contains invalid values. The model still uses the last applied geometry.'
    if edited != {k: model[k] for k in ('joints','tool')}:
        return f"Unapplied edits. The model and solver still use a {model['tool']:g} mm tool. Press Apply geometry & limits."
    return f"Editor matches applied geometry · tool {model['tool']:g} mm."


@app.callback(Output('baseline', 'data'), Output('baseline-message', 'children'),
              Input('save-baseline', 'n_clicks'), Input('clear-baseline', 'n_clicks'), State('model', 'data'), prevent_initial_call=True)
def save_baseline(save, clear, model):
    if ctx.triggered_id == 'clear-baseline':
        return None, 'Baseline cleared.'
    return model, 'Applied design saved in amber. Change limits and Apply to compare.'


@app.callback(Output('download', 'data'), Input('export', 'n_clicks'), State('model', 'data'), prevent_initial_call=True)
def export_model(n, model):
    return dict(content=json.dumps(model, indent=2), filename='arm-geometry.json')


@app.callback(Output('joints', 'data', allow_duplicate=True), Output('tool', 'value', allow_duplicate=True),
              Output('config-message', 'children', allow_duplicate=True), Output('table-enabled','value'), Input('upload', 'contents'), prevent_initial_call=True)
def import_model(contents):
    import base64
    try:
        data = json.loads(base64.b64decode(contents.split(',', 1)[1]))
        model = validate(data['joints'], data['tool'])
        return model['joints'], model['tool'], 'Imported into editor. Press Apply to use this geometry.', ['yes'] if data.get('table',False) else []
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return no_update, no_update, 'Could not import. Choose an exported arm geometry JSON file.', no_update


TARGET_IDS = ['target-' + a for a in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']]


# Keep the actual user camera independently of figure regeneration. Plotly emits
# camera changes for both orbiting and zooming; resize events should not erase it.
app.clientside_callback(
    """function(event, previous) {
        if (!event) return window.dash_clientside.no_update;
        const camera = JSON.parse(JSON.stringify(previous || {}));
        let changed = false;
        for (const [key, value] of Object.entries(event)) {
            if (key === 'scene.camera') return value;
            if (!key.startsWith('scene.camera.')) continue;
            const path = key.slice('scene.camera.'.length).split('.');
            let node = camera;
            for (const part of path.slice(0, -1)) {
                node[part] = node[part] || {};
                node = node[part];
            }
            node[path[path.length - 1]] = value;
            changed = true;
        }
        return changed ? camera : window.dash_clientside.no_update;
    }""",
    Output('workspace-camera', 'data'), Input('scene', 'relayoutData'),
    State('workspace-camera', 'data'), prevent_initial_call=True,
)


@app.callback(*[Output(id, 'value') for id in TARGET_IDS], Input('capture-pose', 'n_clicks'),
              State('model', 'data'), State({'type': 'angle', 'index': ALL}, 'value'), prevent_initial_call=True)
def capture_pose(n, model, angles):
    import warnings
    if len(angles) != len(model['joints']):
        return [no_update] * 6
    p, r, _ = forward(model, angles)
    # Euler angles are nonunique at gimbal lock; SciPy still returns an equivalent rotation.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        euler = Rotation.from_matrix(r).as_euler('xyz', degrees=True)
    return [round(float(v), 6) for v in np.r_[p, euler]]


@app.callback(Output('live-pose', 'children'), Input({'type': 'angle', 'index': ALL}, 'value'),
              Input('model', 'data'), *[Input(id, 'value') for id in TARGET_IDS])
def live_pose(angles, model, x, y, z, roll, pitch, yaw):
    if len(angles) != len(model['joints']):
        return 'Updating arm…'
    p, r, chain = forward(model, angles)
    text = f'Tool XYZ: {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} mm'
    if model.get('table'):
        text += ' | TABLE COLLISION' if np.min(chain[:, 2]) < -1e-6 else ' | Table clear'
    if all(v is not None and np.isfinite(v) for v in (x, y, z, roll, pitch, yaw)):
        text += f'  |  Target error: {np.linalg.norm(p - [x,y,z]):.1f} mm · {float(angular_error(r, orientation(roll,pitch,yaw))):.1f}°'
    return text


@app.callback(Output('solution', 'data'), Output('solve-message', 'children'),
              Input('solve', 'n_clicks'), Input('model', 'data'),
              *[Input(id, 'value') for id in TARGET_IDS], Input('match-orientation', 'value'),
              State({'type': 'angle', 'index': ALL}, 'value'),
              running=[(Output('solve', 'disabled'), True, False)])
def find_pose(n, model, x, y, z, roll, pitch, yaw, match, angles):
    if ctx.triggered_id != 'solve':
        # Target edits must not reset the manually posed arm or the IK starting point.
        return no_update, 'Ready to solve. Tolerances: 2 mm and 2°.'
    values = [x, y, z, roll, pitch, yaw]
    if any(v is None or not np.isfinite(v) for v in values):
        return None, 'Enter all six target values.'
    result = solve(model, [x, y, z], orientation(roll, pitch, yaw) if match else None,
                   angles if len(angles) == len(model['joints']) else None)
    label = 'Pose found' if result['found'] else 'No solution found — inconclusive'
    error = f"Position error {result['distance']:.2f} mm"
    if match:
        error += f" · orientation error {result['angle']:.2f}°"
    if model.get('table'):
        error += f" · lowest centerline z {result['clearance']:.2f} mm"
    detail = 'Last solve succeeded. Move the sliders to explore; live errors are shown below the model.'
    if not result['found']:
        detail = 'Showing the best attempt. No solution found does not prove the target is unreachable.'
        if match:
            position_only = solve(model, [x, y, z], None, result['angles'])
            if position_only['found']:
                detail += ' Position-only check succeeded within these limits. The full orientation requirement was not solved. Uncheck Match full orientation to try that pose.'
            else:
                detail += ' Position-only check also failed within these limits.'
        max_reach = sum(np.linalg.norm([j[k] for k in ('x', 'y', 'z')]) for j in model['joints']) + model['tool']
        if np.linalg.norm([x,y,z]) > max_reach + 2:
            detail += f' This target is outside the chain’s absolute {max_reach:.0f} mm distance bound from the base.'
    return result, html.Div([html.Strong(label), html.Div(error),
                           html.Div(f'Solved target: XYZ ({x:g}, {y:g}, {z:g}) mm · '
                                    + (f'RPY ({roll:g}, {pitch:g}, {yaw:g})°' if match else 'position only')
                                    + f" · applied tool {model['tool']:g} mm"),
                           html.Div('Joint angles: ' + ', '.join(f'J{i+1} {q:.3f}°' for i, q in enumerate(result['angles']))),
                           html.Small(detail)],
                          className='success' if result['found'] else 'warning')


@app.callback(Output('pose-controls', 'children'), Output('model-badge', 'children'),
              Input('model', 'data'), Input('solution', 'data'))
def pose_controls(model, solution):
    try:
        model_changed = ctx.triggered_id == 'model'
    except MissingCallbackContextException:
        model_changed = False
    angles = solution['angles'] if solution and not model_changed and len(solution['angles']) == len(model['joints']) else [np.clip(0, j['lower'], j['upper']) for j in model['joints']]
    controls = []
    for i, (joint, angle) in enumerate(zip(model['joints'], angles)):
        locked = joint['lower'] == joint['upper']
        controls.append(html.Div([
            html.Label(f"J{i + 1} · {joint['name']} · {joint['axis']}"),
            dcc.Slider(id={'type': 'angle', 'index': i}, min=joint['lower'],
                       max=joint['upper'] if not locked else joint['upper'] + 1,
                       step=.1, value=float(np.clip(angle, joint['lower'], joint['upper'])), disabled=locked, updatemode='drag',
                       marks={joint['lower']: f"{joint['lower']:g}°", joint['upper']: f"{joint['upper']:g}°"},
                       tooltip={'placement': 'bottom', 'always_visible': True}),
        ], className='joint-control'))
    active = sum(j['upper'] > j['lower'] for j in model['joints'])
    return controls, f"{len(model['joints'])} JOINTS / {active} ACTIVE DOF"


@app.callback(Output('scene', 'figure'), Output('slice', 'figure'), Output('metrics', 'children'), Output('sampling-note', 'children'), Output('scene-traces', 'data'),
              Input('model', 'data'), Input('baseline', 'data'), Input({'type': 'angle', 'index': ALL}, 'value'),
              *[Input(id, 'value') for id in TARGET_IDS], Input('view-mode', 'value'), Input('quality', 'value'),
              Input('cell-size', 'value'), Input('filter-orientation', 'value'), Input('orientation-tolerance', 'value'),
              Input('plane', 'value'), Input('slice-level', 'value'), State('scene-traces', 'data'),
              State('workspace-camera', 'data'))
def update_views(model, baseline, angles, x, y, z, roll, pitch, yaw, mode, power, size, filter_, tolerance, plane, level, trace_map=None, camera=None):
    if any(v is None or not np.isfinite(v) for v in [x, y, z, roll, pitch, yaw, size, tolerance, level]):
        return no_update, no_update, no_update, 'Complete all numeric fields to update the view.', no_update
    if not 10 <= size <= 500 or not 1 <= tolerance <= 180:
        return no_update, no_update, no_update, 'Cell size must be 10–500 mm; orientation tolerance must be 1–180°.', no_update
    if len(angles) != len(model['joints']):
        angles = [np.clip(0, j['lower'], j['upper']) for j in model['joints']]
    r = orientation(roll, pitch, yaw)
    try:
        dragging = isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get('type') == 'angle'
    except MissingCallbackContextException:
        dragging = False
    if dragging and trace_map:
        # Patch only arm/frame coordinates; retain the expensive workspace and camera.
        fresh = scene(model, angles, np.empty((0, 3)), None, [x,y,z], r, 'hidden', size)
        moving = {trace.name: trace for trace in fresh.data if trace.name == 'Arm' or trace.name.startswith('Tool ')}
        patch = Patch()
        if camera:
            patch['layout']['scene']['camera'] = camera
        for name, i in trace_map.items():
            if name in moving:
                updated = moving[name]
                for axis in 'xyz':
                    patch['data'][i][axis] = np.asarray(getattr(updated, axis)).tolist()
        return patch, no_update, no_update, no_update, no_update
    points = workspace(model, power, r if filter_ else None, tolerance)
    base = workspace(dict(baseline, table=model.get('table',False)), power, r if filter_ else None, tolerance) if baseline else None
    # Prevent very large imported geometries from creating huge dense 2D grids.
    extent = max([np.max(np.abs(p)) if len(p) else 0 for p in [points] + ([base] if base is not None else [])])
    size = max(size, float(np.ceil(extent / 100)))
    occupied = len(cells(points, size))
    metrics = [html.Div([html.Strong(f'{len(points):,}'), html.Span('matching samples' if filter_ else 'sampled poses')]),
               html.Div([html.Strong(f'{occupied:,}'), html.Span('occupied cells')]),
               html.Div([html.Strong(f'{size:g} mm'), html.Span('effective cell size')])]
    note = 'Sampled approximation. Empty cells are unverified, not proven unreachable. Cell size changes the visual resolution.'
    if filter_:
        note += f' Full tool orientations within {tolerance:g}° of the target are retained. Tight tolerances can leave very few samples; use Find a pose to check a specific target.'
    figure = scene(model, angles, points, base, [x, y, z], r, mode, size)
    if camera:
        figure.update_layout(scene_camera=camera)
    return (figure, slice_view(points, base, plane, level, size, [x, y, z]), metrics, note,
            {trace.name: i for i, trace in enumerate(figure.data)})


@app.callback(Output('design-lengths','data'), Input('model','data'), State('design-lengths','data'))
def sync_design_lengths(model, previous):
    rows = length_rows(model)
    old = {r['index']: r for r in previous or []}
    for row in rows:
        if row['index'] in old and row['name'] == old[row['index']]['name']:
            row.update({key:old[row['index']][key] for key in ('optimize','minimum','maximum')})
    return rows


@app.callback(Output('design-targets','data'), Input('save-design-target','n_clicks'),
              State('design-slot','value'), State('design-targets','data'),
              *[State(id,'value') for id in TARGET_IDS], prevent_initial_call=True)
def save_design_target(n, slot, rows, *values):
    if any(v is None or not np.isfinite(v) for v in values):
        return no_update
    rows[slot].update(dict(zip(('x','y','z','roll','pitch','yaw'), values)))
    return rows


@app.callback(*[Output(id,'value',allow_duplicate=True) for id in TARGET_IDS],
              Input('load-design-target','n_clicks'), State('design-slot','value'),
              State('design-targets','data'), prevent_initial_call=True)
def load_design_target(n, slot, rows):
    return [rows[slot][k] for k in ('x','y','z','roll','pitch','yaw')]


@app.callback(Output('design-result','data'), Output('design-message','children'),
              Input('optimize-design','n_clicks'), State('model','data'), State('design-targets','data'),
              State('design-lengths','data'), State('design-effort','value'), prevent_initial_call=True,
              running=[(Output('optimize-design','disabled'),True,False)])
def run_design(n, model, targets, settings, effort):
    try:
        result = optimize_lengths(model, targets, settings, restarts=effort)
    except ValueError as error:
        return None, str(error)
    if not result['found']:
        return None, result['message']
    result['source'] = dict(model=model, targets=targets, settings=settings)
    lines = [html.Strong(f"Verified design · selected length {result['original_length']:.1f} → {result['length']:.1f} mm"),
             html.Div('Shortest verified candidate found; constrained search converged.' if result['converged'] else 'Verified candidate found, but length minimization did not converge.')]
    for link in result['lengths']:
        lines.append(html.Div(f"J{link['index']+1} offset: {link['before']:.2f} → {link['after']:.2f} mm"))
    for i, check in enumerate(result['checks']):
        lines.append(html.Div(f"Pose {i+1}: {check['distance']:.3f} mm / {check['angle']:.3f}° error"
                              + (f" · min z {check['clearance']:.3f} mm" if model.get('table') else '')))
        lines.append(html.Small('Angles: ' + ', '.join(f'{q:.2f}°' for q in check['angles'])))
    lines.append(html.P('Apply the design, then load a pose slot and use Find a pose to inspect it. The previous geometry is saved as the baseline.', className='note'))
    return result, html.Div(lines, className='success')


@app.callback(Output('apply-design','disabled'), Output('design-stale','children'),
              Input('design-result','data'), Input('model','data'), Input('design-targets','data'), Input('design-lengths','data'))
def design_status(result, model, targets, settings):
    if not result:
        return True, 'Choose links and bounds, then run the search.'
    if result['source'] != dict(model=model, targets=targets, settings=settings):
        return True, 'Settings changed since this result was calculated (or the result was applied). Run again before applying another design.'
    return False, 'All target poses verified with the applied joint limits and table setting.'


@app.callback(Output('model','data',allow_duplicate=True), Output('joints','data',allow_duplicate=True),
              Output('tool','value',allow_duplicate=True), Output('baseline','data',allow_duplicate=True),
              Output('config-message','children',allow_duplicate=True), Input('apply-design','n_clicks'),
              State('design-result','data'), State('model','data'), State('design-targets','data'),
              State('design-lengths','data'), prevent_initial_call=True)
def apply_design(n, result, model, targets, settings):
    if not result or result['source'] != dict(model=model, targets=targets, settings=settings):
        return no_update, no_update, no_update, no_update, 'Design result is stale. Run the optimizer again.'
    candidate = result['model']
    return candidate, candidate['joints'], candidate['tool'], model, 'Verified optimized lengths applied. Previous geometry saved as baseline.'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=False)
