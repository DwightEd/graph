import unittest

from sample_analysis import run_windows


class SampleAnalysisTests(unittest.TestCase):
    def test_nearby_runs_keep_clean_windows_disjoint(self):
        windows = run_windows(
            134,
            [(81, 84), (85, 87)],
            pre_window=10,
            post_window=10,
        )
        self.assertEqual(windows[0].as_dict()["clean_pre"], [71, 81])
        self.assertEqual(windows[0].as_dict()["clean_post"], [84, 85])
        self.assertEqual(windows[1].as_dict()["clean_pre"], [84, 85])
        self.assertEqual(windows[1].as_dict()["clean_post"], [87, 97])

    def test_invalid_run_is_rejected(self):
        with self.assertRaises(ValueError):
            run_windows(10, [(9, 11)])


if __name__ == "__main__":
    unittest.main()
