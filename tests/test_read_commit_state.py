import numpy as np
from read_commit_state import delayed_state


def test_delay_state_causal_no_backfill():
    h=np.array([2.,1.,4.,3.,8.,1.,2.])
    e=np.array([0,1,0,0,0,0,0],bool)
    np.testing.assert_equal(delayed_state(h,e,2),[2,1,4,4,4,4,4])
    np.testing.assert_equal(delayed_state(h[:4],e[:4],2),[2,1,4,4])


def test_new_read_resets_state():
    h=np.array([2.,1.,4.,3.,8.,1.,2.])
    e=np.array([0,1,0,0,0,1,0],bool)
    np.testing.assert_equal(delayed_state(h,e,2),[2,1,4,4,4,1,2])


def test_without_event_native_entropy():
    h=np.array([2.,1.,4.])
    np.testing.assert_equal(delayed_state(h,np.zeros(3,bool),8),h)
