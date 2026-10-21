import unittest
import numpy as np
from experiment_loso import decompose


class DecompositionTests(unittest.TestCase):
    def test_four_exclusive_outcomes(self):
        classes=['30_fist','30_open','60_fist','60_open']
        r=decompose(np.array([0,0,0,0]),np.array([0,2,1,3]),classes)
        self.assertEqual([r[k] for k in ['both_correct','gesture_correct_angle_wrong',
                                       'angle_correct_gesture_wrong','both_wrong']],[1,1,1,1])
        self.assertEqual(r['accuracy'],.25)
        self.assertEqual(r['gesture_accuracy'],.5)
        self.assertEqual(r['angle_accuracy'],.5)

    def test_label_order_does_not_define_semantics(self):
        r=decompose(np.array([0,0,0,0]),np.array([0,1,3,2]),
                    ['60_open','30_open','30_fist','60_fist'])
        self.assertEqual(r['gesture_correct_angle_wrong'],1)
        self.assertEqual(r['angle_correct_gesture_wrong'],1)
        self.assertEqual(r['both_wrong'],1)


if __name__=='__main__':
    unittest.main()
