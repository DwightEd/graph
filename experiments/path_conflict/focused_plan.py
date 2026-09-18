"""Registered follow-up to the uploaded pilot. No new head selection by outcomes."""

from dataclasses import dataclass
from .native import Intervention


TARGET_HEADS = ((22, 28), (23, 6), (31, 14), (31, 21))
SOURCE_ROLES = {
    '14315_headwear_scope': {
        'scope': ['Only the Inca could wear'],
        'supported_value': ['Both men and women used cap. On their cap they tied a folded piece of cloth.'],
        'value_source': ['a headdress with his special fringe of gold and feathers.'],
    },
    '14375_onion_stage': {
        'scope': ['Remove the bratwurst from the beer mixture;'],
        'supported_value': ['reduce heat to low, and continue cooking the onions.'],
        'value_source': ['Reduce heat to medium and cook another 10 to 12 minutes.'],
    },
}
GROUPS = ('scope', 'supported_value', 'value_source', 'query_self',
          'recent_history', 'remote_history', 'other_prompt')
EVIDENCE = ('scope', 'supported_value')
HISTORY = ('query_self', 'recent_history', 'remote_history')


@dataclass(frozen=True)
class Trial:
    name: str
    actions: tuple = ()
    kind: str = 'cut'
    restore_layer: int | None = None
    restore_site: str = 'heads'
    restore_heads: tuple = ()
    parents: tuple = ()


def message_action(layer, head, groups, dose=1.0, operation='cut', seed=0):
    return Intervention(layer, tuple(groups), 'query', (head,), operation, seed, dose)


def focused_trials(random_repeats=3):
    trials = [Trial('full', kind='baseline')]
    for layer, head in TARGET_HEADS:
        for group in GROUPS:
            action = message_action(layer, head, (group,))
            trials.append(Trial(f'L{layer}H{head}_{group}', (action,)))
    early = tuple(message_action(layer, head, EVIDENCE) for layer, head in TARGET_HEADS[:2])
    late = (message_action(31, 14, HISTORY),)
    trials.extend([
        Trial('early_22', (early[0],)), Trial('early_23', (early[1],)),
        Trial('early_joint', early, 'joint', parents=('early_22', 'early_23')),
        Trial('late_history', late),
        Trial('early_and_history', early + late, 'joint', parents=('early_joint', 'late_history')),
        Trial('null_zero_dose', (message_action(22, 28, EVIDENCE, 0.0),), 'null'),
    ])
    for group in ('scope', 'supported_value'):
        actions = tuple(message_action(layer, head, (group,)) for layer, head in TARGET_HEADS[:2])
        trials.append(Trial('early_joint_' + group, actions, 'joint',
                            parents=(f'L22H28_{group}', f'L23H6_{group}')))
    for name, actions in (('early_22', early[:1]), ('early_23', early[1:]), ('early_joint', early)):
        half = tuple(message_action(item.layer, item.heads[0], item.groups, .5) for item in actions)
        trials.append(Trial(name + '_half', half, 'dose', parents=(name,)))
    trials.extend(control_trials(early, late, random_repeats))
    trials.extend(restore_trials(early, late))
    return trials


def control_trials(early, late, repeats):
    trials = []
    for name, actions in (('early_joint', early), ('late_history', late)):
        for seed in range(repeats):
            random = tuple(message_action(item.layer, item.heads[0], item.groups, operation='random', seed=seed)
                           for item in actions)
            trials.append(Trial(f'{name}_random_{seed}', random, 'random', parents=(name,)))
    # Fixed adjacent head IDs, not selected by their effects in the new run.
    for layer, target in TARGET_HEADS:
        groups = EVIDENCE if target != 14 else HISTORY
        for head in (target - 1, target + 1):
            action = message_action(layer, head, groups)
            trials.append(Trial(f'control_L{layer}H{head}', (action,), 'head_control'))
    return trials


def restore_trials(early, late):
    trials = []
    for layer in range(27, 32):
        trials.append(Trial(f'early_restore_mlp_{layer}', early, 'restore', layer, 'mlp',
                            parents=('early_joint',)))
    for head in (14, 21):
        trials.append(Trial(f'early_restore_L31H{head}', early, 'restore', 31, 'heads', (head,),
                            parents=('early_joint',)))
    trials.append(Trial('history_hold_mlp31', late, 'restore', 31, 'mlp', parents=('late_history',)))
    return trials
