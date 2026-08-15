import unittest

import numpy as np

from attention_graph.aligned_reservoir import AlignedReservoir


class AlignedReservoirTests(unittest.TestCase):
    def test_eviction_keeps_all_blocks_on_the_same_token_rows(self):
        rows = np.arange(7, dtype=np.float32)[:, None]
        reservoir = AlignedReservoir(position_bins=1, size=3, seed=7)
        reservoir.add("fit", {"prompt": rows, "history": rows + 100.0}, np.zeros(7))

        values, bins = reservoir.matrix("fit")
        priority = np.random.default_rng(7).random(7)
        expected = rows[np.argsort(priority, kind="stable")[:3]]

        np.testing.assert_array_equal(bins, np.zeros(3, dtype=np.int16))
        np.testing.assert_array_equal(values["prompt"], expected)
        np.testing.assert_array_equal(values["history"], expected + 100.0)
        self.assertEqual(reservoir.block_names, ("prompt", "history"))
        np.testing.assert_array_equal(reservoir.block("fit", "prompt"), expected)
        np.testing.assert_array_equal(reservoir.bins("fit"), bins)

    def test_batch_partition_does_not_change_the_sample(self):
        rows = np.arange(12, dtype=np.float32)
        blocks = {
            "prompt": np.stack((rows, rows + 1.0), axis=1),
            "history": np.stack((rows + 10.0, rows + 20.0), axis=1),
        }
        position = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
        whole = AlignedReservoir(position_bins=3, size=6, seed=19)
        split = AlignedReservoir(position_bins=3, size=6, seed=19)
        whole.add("fit", blocks, position)
        split.add("fit", {name: value[:5] for name, value in blocks.items()}, position[:5])
        split.add("fit", {name: value[5:] for name, value in blocks.items()}, position[5:])

        whole_values, whole_bins = whole.matrix("fit")
        split_values, split_bins = split.matrix("fit")
        np.testing.assert_array_equal(split_bins, whole_bins)
        for name in blocks:
            np.testing.assert_array_equal(split_values[name], whole_values[name])

    def test_snapshot_restore_keeps_fit_and_calibration_streams_identical(self):
        rows = np.arange(10, dtype=np.float32)[:, None]
        position = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
        direct = AlignedReservoir(position_bins=2, size=4, seed=31)
        direct.add("fit", {"x": rows}, position)
        direct.add("cal", {"x": rows + 50.0}, position)

        interrupted = AlignedReservoir(position_bins=2, size=4, seed=31)
        interrupted.add("fit", {"x": rows[:4]}, position[:4])
        interrupted.add("cal", {"x": rows[:4] + 50.0}, position[:4])
        resumed = AlignedReservoir(position_bins=2, size=4, seed=31).restore(
            interrupted.snapshot()
        )
        resumed.add("fit", {"x": rows[4:]}, position[4:])
        resumed.add("cal", {"x": rows[4:] + 50.0}, position[4:])

        for group in ("fit", "cal"):
            direct_values, direct_bins = direct.matrix(group)
            resumed_values, resumed_bins = resumed.matrix(group)
            np.testing.assert_array_equal(resumed_bins, direct_bins)
            np.testing.assert_array_equal(resumed_values["x"], direct_values["x"])
        direct_state = direct.snapshot()
        resumed_state = resumed.snapshot()
        for group in ("fit", "cal"):
            np.testing.assert_array_equal(
                resumed_state["priorities"][group], direct_state["priorities"][group]
            )
            np.testing.assert_array_equal(
                resumed_state["filled"][group], direct_state["filled"][group]
            )

    def test_checkpoint_snapshot_and_restore_can_reuse_array_storage(self):
        rows = np.arange(6, dtype=np.float32)[:, None]
        reservoir = AlignedReservoir(position_bins=1, size=4, seed=5)
        reservoir.add("fit", {"x": rows}, np.zeros(6))
        checkpoint = reservoir.snapshot(copy_arrays=False)
        restored = AlignedReservoir(position_bins=1, size=4, seed=5).restore(
            checkpoint, copy_arrays=False
        )

        self.assertIs(
            restored.snapshot(copy_arrays=False)["values"]["fit"]["x"],
            checkpoint["values"]["fit"]["x"],
        )
        np.testing.assert_array_equal(
            restored.block("fit", "x"), reservoir.block("fit", "x")
        )


if __name__ == "__main__":
    unittest.main()
