"""Label-free local-to-remote events, including remote response carriers."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..message_lineage import attention_chunks
from .cache import read_trace


@dataclass(frozen=True)
class EventConfig:
    window: int = 10
    gain: float = .10
    local_floor: float = .50

    def __post_init__(self):
        if self.window < 1 or not 0 < self.gain <= 1 or not 0 <= self.local_floor <= 1:
            raise ValueError('window>=1, 0<gain<=1 and 0<=local_floor<=1 required')


def row_change(current, previous, query, special, window):
    """Both rows use the CURRENT source partition, preventing window-aging events.

    Inputs are full native attention rows [head, source]. Normalize only for
    descriptive comparisons; neither model attention nor seed messages change.
    """
    source = np.arange(len(special))
    ordinary = ~special & (source <= query)
    remote = ordinary & (query-source > window)
    a, b = current * ordinary, previous * ordinary
    den_a, den_b = a.sum(-1, keepdims=True), b.sum(-1, keepdims=True)
    a = np.divide(a, den_a, out=np.full_like(a, np.nan), where=den_a > 0)
    b = np.divide(b, den_b, out=np.full_like(b, np.nan), where=den_b > 0)
    gain = (a-b) * remote
    positive_gain = np.maximum(gain, 0)
    positive_total = positive_gain.sum(-1)
    gain_distribution = np.divide(
        positive_gain,
        positive_total[:, None],
        out=np.zeros_like(positive_gain),
        where=positive_total[:, None] > 0,
    )
    gain_entropy = -np.where(
        gain_distribution > 0,
        gain_distribution * np.log(np.maximum(gain_distribution, 1e-30)),
        0,
    ).sum(-1)
    best = np.argmax(np.nan_to_num(gain, nan=-np.inf), axis=-1)
    positive = gain[np.arange(len(a)), best] > 0
    return dict(remote_mass=(a*remote).sum(-1), local_mass=(a*(ordinary & ~remote)).sum(-1),
                remote_gain=gain.sum(-1), previous_local=(b*(ordinary & ~remote)).sum(-1),
                time_tv=.5*np.abs(a-b).sum(-1),
                remote_positive_gain=positive_total,
                remote_gain_focality=np.divide(
                    positive_gain.max(-1), positive_total,
                    out=np.full_like(positive_total, np.nan), where=positive_total > 0),
                remote_gain_effective_sources=np.where(
                    positive_total > 0, np.exp(gain_entropy), np.nan),
                remote_gain_distance=np.where(
                    positive_total > 0,
                    (gain_distribution * (query-source)).sum(-1), np.nan),
                peak_source=np.where(positive, best, -1))


def detect_layer(chunks, rows, special, start, config):
    """Streaming chunks [H,Q,N]; event construction has no label argument."""
    output, previous = None, None
    for begin, end, attention in chunks:
        a = attention.detach().cpu().numpy() if torch.is_tensor(attention) else attention
        if output is None:
            output = {name: np.full((a.shape[0], len(rows)), np.nan, np.float32)
                      for name in ('remote_mass','local_mass','remote_gain','previous_local','time_tv')}
            for name in (
                'remote_positive_gain', 'remote_gain_focality',
                'remote_gain_effective_sources', 'remote_gain_distance',
            ):
                output[name] = np.full((a.shape[0], len(rows)), np.nan, np.float32)
            output['peak_source'] = np.full((a.shape[0], len(rows)), -1, np.int32)
            output['event'] = np.zeros((a.shape[0], len(rows)), bool)
        for i, query in enumerate(rows[begin:end], begin):
            row = a[:, i-begin]
            if previous is not None:
                values = row_change(row, previous, int(query), special, config.window)
                for name, value in values.items(): output[name][:, i] = value
                # b is an ordinary RESPONSE carrier with at least one later observed token.
                valid = query >= start and query+1 < len(special) and not special[query] and not special[rows[i-1]]
                output['event'][:, i] = (valid & (values['remote_gain'] >= config.gain)
                                         & (values['previous_local'] >= config.local_floor))
            previous = row.copy()
    if output is None: raise ValueError('no attention rows')
    return output


@torch.inference_mode()
def scan(path, config=EventConfig(), *, chunk=8, device='cpu', progress=None):
    path = Path(path)
    trace = read_trace(path)
    layers = []
    with ExitStack() as stack:
        qk = stack.enter_context(np.load(path.with_suffix('.qk.npz'), allow_pickle=False))
        history = stack.enter_context(np.load(path.with_suffix('.history.npz'), allow_pickle=False))
        count = len([k for k in qk.files if k.startswith('query_')])
        for layer in range(count):
            if progress: progress(f'events L{layer+1}/{count}')
            chunks = attention_chunks(trace, qk, history, layer, device, chunk)
            layers.append(detect_layer(chunks, trace['row_position'], trace['special_mask'],
                                       int(trace['response_start']), config))
    result = {k: np.stack([d[k] for d in layers]) for k in layers[0]}
    # Stable event IDs depend only on native layer/head/row coordinates.
    result['event_index'] = np.argwhere(result['event']).astype(np.int32)
    result.update({k: trace[k] for k in ('token_ids','token_text','row_position','response_start',
                                       'special_mask','capture_special_mask','evidence_mask',
                                       'predictor_logprob') if k in trace})
    result['reanchor_metric_schema'] = np.array(1)
    result['labels_used'] = np.array(False)
    import json
    result['event_config'] = np.array(json.dumps(asdict(config), sort_keys=True))
    return result
