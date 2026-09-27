"""Focused QA wiring, with deterministic fake generation rather than model claims."""
import json
import sys

import numpy as np
import pytest

from experiments.native_support.claim_verification import focused


@pytest.mark.parametrize('valid_answers', [True, False])
def test_source_only_reconstruction_and_full_token_fallback(tmp_path, monkeypatch, valid_answers):
    record = dict(id='one',group='regression',source='source fact',text='A B',
        token_ids=[1,2,3],offsets=[[0,1],[1,2],[2,3]],
        sentences=[dict(start=0,stop=3,text='A B')])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(records=[record],model='fake')))
    (tmp_path/'focused_protocol.json').write_text(json.dumps(dict(ids=['one'])))
    folder=tmp_path/'responses/one';folder.mkdir(parents=True)
    (folder/'isolated_audit.json').write_text(json.dumps(dict(direct_scores=[-2.])))
    question=dict(sentence=0,quote='B',question='What is the value?')

    class Reader:
        def __init__(self, model, batch_size):
            self.calls=0
        def generate(self,messages,max_new_tokens):
            self.calls+=1
            payload=json.loads(messages[0][1]['content'])
            if self.calls==1:
                assert 'source' not in payload
                answer=json.dumps(dict(questions=[question]))
            else:
                assert payload==dict(source='source fact',questions=[dict(id=0,question='What is the value?')])
                answer=json.dumps(dict(answers=[dict(id=0,answer='C')])) if valid_answers else 'invalid'
            return [dict(text=answer,reached_limit=False)]
        def margins(self,messages):
            if not valid_answers:
                assert messages==[]
                return []
            assert len(messages)==2
            assert 'A = supported' in messages[0][0]['content']
            assert 'A = unsupported' in messages[1][0]['content']
            assert json.loads(messages[0][1]['content'])==dict(
                question='What is the value?',proposed_value='B',source_only_answer='C')
            return [6.,-4.]

    monkeypatch.setattr(focused,'Reader',Reader)
    monkeypatch.setattr(sys,'argv',['focused','--output',str(tmp_path)])
    focused.main()
    with np.load(folder/'focused_scores.npz') as saved:
        np.testing.assert_array_equal(saved['token_id'],[1,2,3])
        np.testing.assert_allclose(saved['focused_reconstruction'],[-2.,-2.,5. if valid_answers else -2.])
    completed=json.loads((tmp_path/'focused_completed.json').read_text())
    assert completed['ids']==['one']
