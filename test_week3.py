import unittest
import numpy as np
from week3 import trial_folds, holm, error_summary

class Week3Tests(unittest.TestCase):
    def test_folds_keep_trials_intact_and_cover_once(self):
        groups=np.repeat([f't{i:03d}' for i in range(80)],19)
        y=np.repeat(np.repeat(np.arange(8),10),19)
        seen=[]
        for _,tr,va in trial_folds(y,groups):
            self.assertTrue(set(groups[tr]).isdisjoint(groups[va]))
            self.assertEqual(len(np.unique(y[va])),8)
            self.assertTrue(np.all(np.unique(groups[va],return_counts=True)[1]==19))
            seen.extend(va)
        np.testing.assert_array_equal(np.sort(seen),np.arange(len(y)))

    def test_inconsistent_trial_labels_rejected(self):
        with self.assertRaises(ValueError):list(trial_folds(np.array([0,1]),np.array(['a','a'])))

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])

    def test_error_types_use_labels_not_adjacent_cells(self):
        classes=['30_fist','30_open','60_fist','60_open']
        result=error_summary(np.array([0,1,2,3]),np.array([2,0,1,3]),classes)
        self.assertEqual(result['error_types'],dict(same_gesture_wrong_angle=1,same_angle_wrong_gesture=1,both_wrong=1))

if __name__=='__main__':unittest.main()
