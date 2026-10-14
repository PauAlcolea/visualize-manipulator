import unittest
import numpy as np
from app import app, DEFAULT, update_views, capture_pose, pending_changes
from kinematics import forward, orientation, angular_error
from views import cells


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.server.test_client()

    def test_layout_and_dependencies(self):
        for route in ('/', '/_dash-layout', '/_dash-dependencies', '/assets/style.css'):
            with self.client.get(route) as response:
                self.assertEqual(response.status_code, 200, route)

    def test_solver_callback_round_trip(self):
        key = next(k for k in app.callback_map if k.startswith('..solution.data'))
        values = [1, DEFAULT, 400, 100, 250, 0, 0, 0, ['yes']]
        inputs = [dict(id=item['id'], property=item['property'], value=value)
                  for item, value in zip(app.callback_map[key]['inputs'], values)]
        state_item = app.callback_map[key]['state'][0]
        response = self.client.post('/_dash-update-component', json={
            'output': key, 'outputs': [{'id': 'solution', 'property': 'data'},
                                      {'id': 'solve-message', 'property': 'children'}],
            'inputs': inputs,
            'state': [dict(id=state_item['id'], property=state_item['property'], value=[0] * 6)],
            'changedPropIds': ['solve.n_clicks'],
        })
        self.assertEqual(response.status_code, 200, response.data)
        result = response.get_json()['response']['solution']['data']
        self.assertTrue(result['found'])

    def test_views_with_empty_orientation_samples(self):
        model = dict(joints=[dict(name='Locked', axis='Z', x=100, y=0, z=0, lower=0, upper=0)], tool=0)
        result = update_views(model, DEFAULT, [0], 400, 100, 250, 0, 0, 90,
                              'cells', 12, 40, ['yes'], 1, 'XZ', 0)
        self.assertIn('data', result[0].to_json())
        self.assertIn('data', result[1].to_json())

    def test_cells_keep_disconnected_regions(self):
        occupied = cells(np.array([[0, 0, 0], [100, 0, 0]]), 10)
        self.assertEqual(len(occupied), 2)
        self.assertFalse(np.any(occupied[:, 0] == 5))

    def dispatch(self, prefix, values, states, changed):
        key = next(k for k in app.callback_map if k.startswith(prefix))
        meta = app.callback_map[key]
        outputs = meta['output']
        response = self.client.post('/_dash-update-component', json={
            'output': key,
            'outputs': [dict(id=o.component_id, property=o.component_property) for o in outputs],
            'inputs': [dict(id=i['id'], property=i['property'], value=v) for i, v in zip(meta['inputs'], values)],
            'state': [dict(id=i['id'], property=i['property'], value=v) for i, v in zip(meta['state'], states)],
            'changedPropIds': [changed],
        })
        self.assertEqual(response.status_code, 200, response.data)
        return response.get_json()['response']

    def test_target_edit_preserves_manual_pose(self):
        result = self.dispatch('..solution.data', [1, DEFAULT, 400, 100, 250, 0, 0, 0, ['yes']],
                               [[20, -30, 60, 0, 0, 0]], 'target-x.value')
        self.assertNotIn('solution', result)

    def test_capture_current_pose(self):
        q = [20, -30, 60, 15, -40, 80]
        target = capture_pose(1, DEFAULT, q)
        p, r, _ = forward(DEFAULT, q)
        np.testing.assert_allclose(target[:3], p, atol=1e-5)
        self.assertLess(float(angular_error(r, orientation(*target[3:]))), 1e-5)

    def test_drag_only_patches_arm_traces(self):
        values = [DEFAULT, None, [20, 0, 0, 0, 0, 0], 400, 100, 250, 0, 0, 0,
                  'cells', 12, 40, [], 20, 'XY', 240]
        result = self.dispatch('..scene.figure', values,
                               [{'Current workspace': 0, 'Arm': 1, 'Tool X': 3, 'Tool Y': 4, 'Tool Z': 5}, None],
                               '{"index":0,"type":"angle"}.value')
        self.assertEqual(set(result), {'scene'})
        operations = result['scene']['figure']['operations']
        self.assertEqual(len(operations), 12)
        self.assertTrue(all(op['location'][1] in [1, 3, 4, 5] for op in operations))

    def test_joint_change_preserves_camera(self):
        camera = dict(eye=dict(x=.3, y=-.4, z=.2), center=dict(x=.1, y=0, z=0),
                      up=dict(x=0, y=0, z=1), projection=dict(type='perspective'))
        values = [DEFAULT, None, [20, 0, 0, 0, 0, 0], 400, 100, 250, 0, 0, 0,
                  'cells', 12, 40, [], 20, 'XY', 240]
        result = self.dispatch('..scene.figure', values, [{'Arm': 1}, camera],
                               '{"index":0,"type":"angle"}.value')
        operations = result['scene']['figure']['operations']
        camera_patch = next(op for op in operations if op['location'] == ['layout', 'scene', 'camera'])
        self.assertEqual(camera_patch['params']['value'], camera)
        # Full redraws (including solver-driven joint changes) retain the same camera.
        figure = update_views(*values, camera=camera)[0]
        self.assertEqual(figure.layout.scene.camera.to_plotly_json(), camera)

    def test_failure_explains_position_only_success(self):
        model = dict(joints=[dict(name='Locked', axis='Z', x=100, y=0, z=0, lower=0, upper=0)], tool=0)
        result = self.dispatch('..solution.data', [1, model, 100, 0, 0, 0, 0, 90, ['yes']], [[0]], 'solve.n_clicks')
        self.assertFalse(result['solution']['data']['found'])
        self.assertIn('Position-only check succeeded', str(result['solve-message']))

    def test_reported_zero_tool_target_via_solve_callback(self):
        # Match the browser's JSON encoding, which drops .0 on whole numbers.
        model = dict(joints=[{key: int(value) if isinstance(value, float) and value.is_integer() else value
                            for key, value in joint.items()} for joint in DEFAULT['joints']], tool=0)
        for initial in ([0] * 6, [14, -48, 68, 0, -21, 0], [5, -44, 35, -7, 30, 15]):
            result = self.dispatch('..solution.data', [1, model, 400, 100, 250, 0, 0, 0, ['yes']],
                                   [initial], 'solve.n_clicks')
            solved = result['solution']['data']
            self.assertTrue(solved['found'], solved)
            p, r, _ = forward(model, solved['angles'])
            np.testing.assert_allclose(p, [400, 100, 250], atol=.001)
            self.assertLess(float(angular_error(r, orientation(0, 0, 0))), .001)
            self.assertIn('applied tool 0 mm', str(result['solve-message']))

    def test_zero_tool_edit_is_reported_as_pending(self):
        self.assertIn('Unapplied edits', pending_changes(DEFAULT['joints'], 0, DEFAULT))
        self.assertIn('Editor matches', pending_changes(DEFAULT['joints'], 0, dict(joints=DEFAULT['joints'], tool=0)))


if __name__ == '__main__':
    unittest.main()
