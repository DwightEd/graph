"""清楚的主线：读一答 → 配对 → 逐head测量 → 保存；最后汇总。"""

from pathlib import Path

from tqdm import tqdm

from .matching import match_controls
from .measurements import measure_answer
from .report import load_saved, save_answer, summarize


def analyze_answer(inputs, response_id, output, settings):
    answer = inputs.load_answer(response_id)
    pairs = match_controls(answer, settings['position_gap'],
                           settings['repetition_gap'], settings['entropy_gap'], settings['window'])

    channels = []
    if pairs:
        channels = inputs.channels(answer, settings['layers'], settings['heads'])
    channel_ids, readings, counts = measure_answer(answer, pairs, channels, settings['window'])

    return save_answer(output, answer, pairs, channel_ids, readings, counts)


def analyze_dataset(inputs, output, settings, resume=False):
    identities = list(inputs.selected_ids(settings['split'], settings['tasks']))
    if settings['limit']:
        identities = identities[:settings['limit']]

    for response_id in tqdm(identities, desc='Compare labelled spans', unit='answer'):
        path = Path(output) / 'samples' / f'{response_id}.npz'
        if resume and path.is_file():
            load_saved(path)
            continue
        analyze_answer(inputs, response_id, output, settings)

    return summarize(output, settings['bootstrap'])
