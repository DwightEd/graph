import unittest
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    def test_build_command_constructs_builder_and_prints_summary(self) -> None:
        with patch("main.GraphDatasetBuilder") as builder, patch("builtins.print") as output:
            builder.return_value.run.return_value = {"kind": "relation_topk_channels", "count": 1}
            main.main([
                "build", "--cache-dir", "cache/train", "--output-dir", "outputs/train",
                "--device", "cpu", "--limit", "1",
            ])

        config = builder.call_args.args[0]
        self.assertEqual(config.cache_dir, "cache/train")
        self.assertEqual(config.output_dir, "outputs/train")
        self.assertEqual(config.kind, "relation_topk_channels")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.limit, 1)
        builder.return_value.run.assert_called_once_with()
        output.assert_called_once_with("build relation_topk_channels: 1 graphs")


if __name__ == "__main__":
    unittest.main()
