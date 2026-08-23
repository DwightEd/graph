from types import SimpleNamespace

import pytest

from attention_lifecycle import loaded_attention


class Sample:
    def __init__(self):
        self.value = SimpleNamespace(name="attention")
        self.release_calls = 0

    def attention(self):
        return self.value

    def release_attention(self):
        self.release_calls += 1


def test_loaded_attention_releases_after_normal_use():
    sample = Sample()

    with loaded_attention(sample) as attention:
        assert attention is sample.value

    assert sample.release_calls == 1


def test_loaded_attention_releases_when_sample_processing_fails():
    sample = Sample()

    with pytest.raises(RuntimeError, match="failed"):
        with loaded_attention(sample):
            raise RuntimeError("failed")

    assert sample.release_calls == 1
