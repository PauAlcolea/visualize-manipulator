import unittest
import numpy as np
from kinematics import validate, default_joints, forward, solve, orientation, workspace, angular_error


class KinematicsTests(unittest.TestCase):
    def test_default_zero_pose(self):
        model = validate(default_joints(), 80)
        p, r, chain = forward(model, np.zeros(6))
        np.testing.assert_allclose(p, [600, 0, 120])
        np.testing.assert_allclose(r, np.eye(3))
        self.assertEqual(chain.shape, (8, 3))

    def test_local_rotation_then_translation(self):
        model = validate([dict(name='A', axis='Z', x=100, y=0, z=0, lower=-180, upper=180)], 20)
        p, _, _ = forward(model, [90])
        np.testing.assert_allclose(p, [0, 120, 0], atol=1e-10)

    def test_full_pose_recovered(self):
        model = validate(default_joints(), 80)
        p, r, _ = forward(model, [25, -35, 70, 20, -40, 30])
        result = solve(model, p, r)
        self.assertTrue(result['found'], result)
        self.assertLess(result['distance'], 2)
        self.assertLess(result['angle'], 2)
        for q, joint in zip(result['angles'], model['joints']):
            self.assertTrue(joint['lower'] <= q <= joint['upper'])

    def test_locked_joint_orientation_and_limits(self):
        model = validate([dict(name='A', axis='Z', x=100, y=0, z=0, lower=0, upper=0)], 0)
        self.assertTrue(solve(model, [100, 0, 0], None)['found'])
        self.assertFalse(solve(model, [100, 0, 0], orientation(0, 0, 90))['found'])
        self.assertFalse(solve(model, [0, 100, 0], None)['found'])
        np.testing.assert_allclose(workspace(model, 4), np.tile([100, 0, 0], (16, 1)))

    def test_tight_limits_block_known_target(self):
        model = validate([dict(name='A', axis='Z', x=100, y=0, z=0, lower=-10, upper=10)], 0)
        result = solve(model, [0, 100, 0], None, restarts=3)
        self.assertFalse(result['found'])
        self.assertLessEqual(abs(result['angles'][0]), 10)

    def test_orientation_filter(self):
        model = validate([dict(name='A', axis='Z', x=100, y=0, z=0, lower=-180, upper=180)], 0)
        p = workspace(model, 10, orientation(0, 0, 0), 10)
        self.assertGreater(len(p), 0)
        self.assertTrue(np.all(p[:, 0] >= 100 * np.cos(np.deg2rad(10))))
        self.assertAlmostEqual(float(angular_error(orientation(0, 0, 179), orientation(0, 0, -179))), 2)

    def test_invalid_limits(self):
        rows = default_joints()
        rows[0]['lower'] = 200
        with self.assertRaises(ValueError):
            validate(rows, 80)

    def test_integer_limits_allow_fractional_optimizer_steps(self):
        # Regression: integer limit arrays used to truncate the optimizer's angles.
        model = dict(joints=default_joints(), tool=0)
        target_rotation = orientation(0, 0, 0)
        result = solve(model, [400, 100, 250], target_rotation, [5, -44, 35, -7, 30, 15])
        self.assertTrue(result['found'], result)
        self.assertLess(result['distance'], .001)
        self.assertLess(result['angle'], .001)


if __name__ == '__main__':
    unittest.main()
