import unittest
from copy import deepcopy
import numpy as np
from kinematics import default_joints, validate, forward, solve, workspace, orientation, angular_error
from design import optimize_lengths, length_rows
from app import design_status, apply_design, toggle_table


class TableTests(unittest.TestCase):
    def test_above_table_tip_does_not_hide_below_table_elbow(self):
        model = dict(joints=[
            dict(name='down',axis='Y',x=100,y=0,z=0,lower=90,upper=90),
            dict(name='up',axis='Y',x=200,y=0,z=0,lower=-180,upper=-180),
        ], tool=0)
        p,r,chain = forward(model,[90,-180])
        self.assertGreater(p[2], 0)
        self.assertLess(chain[1,2], 0)
        self.assertTrue(solve(model,p,r)['found'])
        model['table'] = True
        self.assertFalse(solve(model,p,r)['found'])
        self.assertEqual(len(workspace(model,4)),0)

    def test_table_contact_and_toggle(self):
        model = dict(joints=[dict(name='flat',axis='Z',x=100,y=0,z=0,lower=0,upper=0)],tool=0)
        table_model = toggle_table(['yes'],model)
        self.assertTrue(solve(table_model,[100,0,0],orientation(0,0,0))['found'])
        self.assertEqual(len(workspace(table_model,4)),16)
        self.assertNotIn('table',model)


class DesignTests(unittest.TestCase):
    def test_three_poses_shorten_and_verify(self):
        model = validate(default_joints(),80)
        model['table'] = True
        rows = [dict(x=x,y=y,z=z,roll=0,pitch=0,yaw=0)
                for x,y,z in [(400,100,250),(300,-200,200),(450,0,150)]]
        settings = length_rows(model)
        before = deepcopy(model)
        result = optimize_lengths(model,rows,settings,restarts=2)
        self.assertTrue(result['found'])
        self.assertLess(result['length'],result['original_length'] - 100)
        self.assertEqual(model,before)
        for target,check in zip(rows,result['checks']):
            p,r,chain = forward(result['model'],check['angles'])
            self.assertLessEqual(np.linalg.norm(p - [target[k] for k in ('x','y','z')]),2)
            self.assertLessEqual(float(angular_error(r,orientation(0,0,0))),2)
            self.assertGreaterEqual(chain[:,2].min(),-1e-6)
            for q,j in zip(check['angles'],model['joints']):
                self.assertTrue(j['lower'] - 1e-6 <= q <= j['upper'] + 1e-6)
        self.assertEqual(result['model']['joints'][0],model['joints'][0])
        self.assertEqual(result['model']['tool'],80)
        result['source'] = dict(model=model,targets=rows,settings=settings)
        self.assertFalse(design_status(result,model,rows,settings)[0])
        changed = deepcopy(rows)
        changed[0]['x'] += 20
        self.assertTrue(design_status(result,model,changed,settings)[0])
        self.assertIn('stale',apply_design(1,result,model,changed,settings)[4])
        self.assertEqual(apply_design(1,result,model,rows,settings)[0],result['model'])

    def test_known_one_link_minimum_and_locked_joint(self):
        model = dict(joints=[dict(name='link',axis='Z',x=200,y=0,z=0,lower=0,upper=0)],tool=0,table=True)
        settings = [dict(index=0,optimize='yes',minimum=20,maximum=300)]
        target = [dict(x=100,y=0,z=0,roll=0,pitch=0,yaw=0)]
        result = optimize_lengths(model,target,settings,restarts=1)
        self.assertTrue(result['found'])
        self.assertAlmostEqual(result['length'],98.01,places=2)

    def test_infeasible_design_is_not_returned(self):
        model = dict(joints=[dict(name='link',axis='Z',x=50,y=0,z=0,lower=0,upper=0)],tool=0)
        settings = [dict(index=0,optimize='yes',minimum=20,maximum=60)]
        target = [dict(x=100,y=0,z=0,roll=0,pitch=0,yaw=0)]
        self.assertFalse(optimize_lengths(model,target,settings,restarts=1)['found'])
        model['table'] = True
        target[0]['z'] = -10
        with self.assertRaisesRegex(ValueError,'below the table'):
            optimize_lengths(model,target,settings,restarts=1)

    def test_invalid_bounds(self):
        model = validate(default_joints(),0)
        settings = length_rows(model)
        settings[1]['maximum'] = settings[1]['minimum']
        with self.assertRaises(ValueError):
            optimize_lengths(model,[dict(x=100,y=0,z=0,roll=0,pitch=0,yaw=0)],settings)


if __name__ == '__main__':
    unittest.main()
