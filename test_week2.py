import unittest
import numpy as np
import pandas as pd
from week2 import td_features, windows, split_trials, filter_emg


class FeatureTests(unittest.TestCase):
    def test_constant_and_alternating_features(self):
        np.testing.assert_allclose(td_features(np.ones((200, 5))),
                                   np.r_[np.ones(10), np.zeros(15)])
        w = np.tile(np.where(np.arange(200) % 2 == 0, -.1, .1)[:, None], (1, 5))
        np.testing.assert_allclose(td_features(w), np.repeat([.1, .1, 39.8, 199, 198], 5))

    def test_threshold_rejects_small_crossings(self):
        w = np.tile(np.where(np.arange(200) % 2 == 0, -.001, .001)[:, None], (1, 5))
        self.assertTrue(np.all(td_features(w)[15:20] == 0))
        self.assertTrue(np.all(td_features(w, 0)[15:20] == 199))
        self.assertTrue(np.all(td_features(w)[20:] == 198))

    def test_window_endpoints(self):
        segment = np.repeat(np.arange(2000)[:, None], 5, axis=1)
        result = windows(segment)
        self.assertEqual(result.shape, (19, 200, 5))
        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[-1, 0, 0], 1800)
        self.assertEqual(result[-1, -1, 0], 1999)

    def test_trial_split_is_stratified_and_disjoint(self):
        frame = pd.DataFrame({'trial_id': [f't{i}' for i in range(1920)],
                              'y': np.repeat(np.arange(8), 240)})
        result = split_trials(frame)
        tr, te = result[result.split == 'train'], result[result.split == 'test']
        self.assertEqual(len(tr), 1536)
        self.assertEqual(len(te), 384)
        self.assertTrue(set(tr.trial_id).isdisjoint(te.trial_id))
        self.assertTrue((te.groupby('y').size() == 48).all())

    def test_filter_attenuates_notch_frequency(self):
        t = np.arange(5000) / 1000
        x = np.tile(np.sin(2 * np.pi * 60 * t)[:, None], (1, 5))
        self.assertLess(filter_emg(x)[1000:-1000].std(), .05 * x.std())


if __name__ == '__main__':
    unittest.main()
