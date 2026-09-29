import numpy as np
from .measure import partition
from .score import restrict_steps
from experiments.context_response.recurrence import propagate


def test_question_partition_keeps_repeated_problem_together():
    rows = [dict(problem=str(i)) for i in range(10)]+[dict(problem='0')]
    splits = partition(rows)
    assert len(splits)==10
    assert list(splits.values()).count('fit')==2
    assert list(splits.values()).count('dev')==2
    assert list(splits.values()).count('evaluation')==6


def test_step_boundary_blocks_high_risk_seed():
    seed = np.array([1.,0.,0.,0.])
    edges = [np.ones(3)]
    steps = np.array([0,0,1,1])
    assert np.all(propagate(seed,edges)[0]==1)
    assert np.array_equal(propagate(seed,restrict_steps(edges,steps))[0],[1,1,0,0])


def test_first_error_does_not_label_all_later_steps():
    from .evaluate import step_label
    assert [step_label(i,2) for i in range(5)]==[0,0,1,-1,-1]
    assert [step_label(i,-1) for i in range(5)]==[0]*5


def test_attention_row_precedes_predicted_token():
    from .measure import measure_heads
    attention = np.broadcast_to(np.eye(12,dtype=np.float32),(1,32,12,12)).copy()
    raw, pairs, source, error = measure_heads(attention,3)
    assert len(raw)==9
    assert np.all(source[:,0]==1)
    assert np.all(source[:,1:]==0)
    assert all(pairs[f'head_{lag}'].shape==(32,9-lag) for lag in range(1,9))
    assert error==0
