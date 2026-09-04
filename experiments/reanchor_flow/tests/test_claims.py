from experiments.reanchor_flow.claims import sentence_boundaries


class Tokenizer:
    def decode(self, token, **kwargs):
        return {1: "Hello", 2: ".", 3: " Next"}[token[0]]


def test_sentence_boundaries_are_references_after_punctuation():
    boundary = sentence_boundaries(Tokenizer(), [1, 2, 3], response_start=0)
    assert boundary.tolist() == [0, 2]
