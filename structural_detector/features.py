"""Every feature at t depends only on measurements at positions <= t."""
import numpy as np

NAMES=['entropy','negative_margin','source_mass','remote_history_excl16_mass','source_head_std',
       'remote_history_excl16_minus_source','entropy_change','remote_history_excl16_minus_source_change','recent_entropy',
       'entropy_x_remote_history_excl16_minus_source','recent_entropy_x_remote_history_excl16_minus_source']
ARMS={'instant':[0,1,2,3,4,5], 'combined':list(range(11)),
      'without_entropy':[2,3,4,5,7], 'without_attention':[0,1,6,8]}


def features(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all():raise ValueError('five finite native columns required')
    h,margin,source,remote,spread=x.T;d=remote-source
    dh=np.r_[0.,np.diff(h)];dd=np.r_[0.,np.diff(d)]
    memory=np.zeros(len(h))
    for t in range(len(h)):
        # Earlier uncertainty, with no gold onset and no future response length.
        past=np.arange(max(0,t-8),t)
        if len(past):memory[t]=np.max(h[past]*np.exp(-(t-past)/4.))
    return np.column_stack([h,margin,source,remote,spread,d,dh,dd,memory,h*d,memory*d])


def select_sources(rows,seed=20260914):
    import hashlib
    train={r['source_id'] for r in rows if r['official_split']=='train'}
    test={r['source_id'] for r in rows if r['official_split']=='test'}
    if train & test:raise ValueError('official source overlap')
    ordered=sorted(train,key=lambda s:hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    cutoff=int(.8*len(ordered));fit=set(ordered[:cutoff]);cal=set(ordered[cutoff:])
    return {r['id']:('test' if r['source_id'] in test else 'train' if r['source_id'] in fit else 'calibration') for r in rows}
