"""All-head prompt/history routing and signed native writes at audited queries."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.path_conflict.native import request_layer_attention
from experiments.path_conflict.operators import grouped_values

GROUPS = ('source', 'evidence', 'constraint', 'other_source', 'instruction',
          'local_history', 'remote_history', 'claim_prefix')
FIELDS = ('attention_mass', 'write_norm', 'local_logp_projection', 'residual_cosine')


def group_masks(item, positions, length, device):
    prompt = len(item['prompt_ids'])
    keys = torch.arange(length, device=device)
    query = torch.as_tensor(positions, device=device)+prompt-1
    masks = []
    for group in GROUPS:
        if group == 'local_history':
            mask = (keys[None] >= prompt) & (keys[None] <= query[:, None]) & (keys[None] > query[:, None]-16)
        elif group == 'remote_history':
            mask = (keys[None] >= prompt) & (keys[None] <= query[:, None]-16)
        elif group == 'claim_prefix':
            starts = np.full(len(positions), length, dtype=int)
            for start, stop in item['spans']:
                belongs = (np.asarray(positions) >= start) & (np.asarray(positions) < stop+4)
                starts[belongs] = prompt+start
            mask = (keys[None] >= torch.as_tensor(starts, device=device)[:, None]) & (keys[None] <= query[:, None])
        else:
            selected = item['groups'][group]
            mask = torch.zeros((len(positions), length), device=device, dtype=torch.bool)
            mask[:, selected] = True
            mask &= keys[None] <= query[:, None]
        masks.append(mask)
    return torch.stack(masks, 1)


def local_logp_gradient(model, residual, targets):
    """Analytic RMSNorm+unembedding log-p gradient; excludes subsequent layers."""
    state = residual.float()
    norm = model.model.norm
    scale = (state.square().mean(-1, keepdim=True)+norm.variance_epsilon).sqrt()
    normalized = state/scale*norm.weight.float()
    logits = F.linear(normalized.to(model.dtype), model.lm_head.weight).float()
    probability = logits.softmax(-1)
    expectation = torch.zeros_like(state)
    for start in range(0, probability.shape[-1], 4096):
        expectation += probability[:, start:start+4096] @ model.lm_head.weight[start:start+4096].float()
    direction = (model.lm_head.weight[targets].float()-expectation)*norm.weight.float()
    gradient = direction/scale-state*(direction*state).mean(-1, keepdim=True)/scale.pow(3)
    return gradient


class AuditCapture:
    def __init__(self, model, item, positions=None):
        self.model, self.item, self.positions = model, item, positions
        self.handles, self.values, self.residuals = [], {}, {}
        self.mass, self.stats, self.attention, self.key_projection = [], [], [], []
        self.states, self.layer_updates, self.errors = [], [], []
        self.pre_states, self.attn_states, self.site_readouts, self.mlp_stats = [], [], [], []
        self.attn_residuals, self.mlp_outputs = {}, {}
        self.switches, self.previous_attention = [], []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(request_layer_attention, with_kwargs=True))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self.save_values(index)))
            self.handles.append(layer.input_layernorm.register_forward_pre_hook(self.save_residual(index)))
            self.handles.append(layer.self_attn.register_forward_hook(self.observe(index)))
            self.handles.append(layer.mlp.register_forward_hook(self.save_mlp(index)))
            self.handles.append(layer.register_forward_hook(self.save_layer(index)))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def save_values(self, index):
        def hook(module, inputs, output):
            self.values[index] = output.detach()
        return hook

    def save_residual(self, index):
        def hook(module, inputs):
            if self.positions is not None:
                queries = np.asarray(self.positions)+len(self.item['prompt_ids'])-1
                self.residuals[index] = inputs[0][0, queries].detach()
                self.pre_states.append(self.residuals[index].float().cpu().numpy())
        return hook

    def save_mlp(self, index):
        def hook(module, inputs, output):
            if self.positions is not None:
                queries = np.asarray(self.positions)+len(self.item['prompt_ids'])-1
                self.mlp_outputs[index] = output[0, queries].detach()
        return hook

    def save_layer(self, index):
        def hook(module, inputs, output):
            if self.positions is not None:
                state = output[0] if isinstance(output, tuple) else output
                queries = np.asarray(self.positions)+len(self.item['prompt_ids'])-1
                post = state[0, queries]
                self.states.append(post.float().cpu().numpy())
                pre = torch.tensor(self.pre_states[index], device=post.device, dtype=post.dtype)
                after_attention = self.attn_residuals.pop(index)
                targets = torch.tensor(np.asarray(self.item['answer_ids'])[self.positions], device=post.device)
                sites = []
                for residual in (pre, after_attention, post):
                    lp = self.model.lm_head(self.model.model.norm(residual)).float().log_softmax(-1)
                    sites.append(torch.stack((-(lp.exp()*lp).sum(-1),
                        -lp.gather(-1,targets[:,None])[:,0],residual.float().norm(dim=-1)),-1))
                self.site_readouts.append(torch.stack(sites,1).cpu().numpy())
                mlp = self.mlp_outputs.pop(index).float()
                gradient = local_logp_gradient(self.model,post,targets)
                self.mlp_stats.append(torch.stack((mlp.norm(dim=-1),(mlp*gradient).sum(-1),
                    (mlp*post.float()).sum(-1)/(mlp.norm(dim=-1)*post.float().norm(dim=-1)).clamp_min(1e-20)),-1).cpu().numpy())
        return hook

    def observe(self, index):
        def hook(module, inputs, output):
            changed, attention = output[:2]
            prompt = len(self.item['prompt_ids'])
            config = self.model.config
            values = grouped_values(self.values.pop(index), config.num_attention_heads, config.num_key_value_heads)[0].float()
            if self.positions is None:
                # Full trajectory, no gold onset used in automatic candidate selection.
                weights = attention[0, :, prompt-1:].float()
                positions = np.arange(weights.shape[1])
                masks = group_masks(self.item, positions, weights.shape[-1], weights.device)
                mass = torch.einsum('htk,tgk->htg', weights, masks.float())
                self.mass.append(mass.cpu().numpy())
                if 'special_ids' in self.item:
                    from .audit_reanchors import switch_measures
                    self.switches.append(switch_measures(attention,self.item))
                return
            positions = np.asarray(self.positions)
            queries = positions+prompt-1
            weights = attention[0, :, queries].float()
            residual = self.residuals.pop(index)+changed[0, queries]
            self.attn_residuals[index] = residual
            self.attn_states.append(residual.float().cpu().numpy())
            masks = group_masks(self.item, positions, weights.shape[-1], weights.device)
            components = torch.einsum('htk,tgk,hkd->htgd', weights, masks.float(), values)
            width = values.shape[-1]
            blocks = module.o_proj.weight.float().view(-1, config.num_attention_heads, width).permute(1, 0, 2)
            messages = torch.einsum('hod,htgd->htgo', blocks, components)
            targets = torch.tensor(np.asarray(self.item['answer_ids'])[positions], device=weights.device)
            gradient = local_logp_gradient(self.model, residual, targets)
            projected_direction = torch.einsum('to,hod->htd', gradient, blocks)
            key_projection = weights*torch.einsum('htd,hkd->htk', projected_direction, values)
            projection = torch.einsum('htgo,to->htg', messages, gradient)
            norms = messages.norm(dim=-1)
            cosine = torch.einsum('htgo,to->htg', messages, residual.float())/(norms*residual.float().norm(dim=-1)[None,:,None]).clamp_min(1e-20)
            mass = torch.einsum('htk,tgk->htg', weights, masks.float())
            self.stats.append(torch.stack((mass, norms, projection, cosine), -1).cpu().numpy())
            self.attention.append(weights.cpu().to(torch.float16).numpy())
            previous = torch.stack([attention[0,:,np.maximum(queries-i,0)].float() for i in (1,2,3)]).mean(0)
            self.previous_attention.append(previous.cpu().to(torch.float16).numpy())
            self.key_projection.append(key_projection.cpu().to(torch.float16).numpy())
            all_components = weights @ values
            total = torch.einsum('hod,htd->to', blocks, all_components)
            expected = changed[0, queries].float()
            if module.o_proj.bias is not None:
                total += module.o_proj.bias.float()
            self.errors.append(float(((total-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-20)).max()))
        return hook


def capture_audit(model, item, positions=None):
    ids = item['prompt_ids']+item['answer_ids'][:-1]
    tensor = torch.tensor([ids], device=model.device)
    with torch.inference_mode(), AuditCapture(model, item, positions) as run:
        result = model.model(tensor, use_cache=False, output_attentions=False)
        hidden = result.last_hidden_state[0, len(item['prompt_ids'])-1:]
        entropy, surprisal = [], []
        for start in range(0, len(hidden), 32):
            lp = model.lm_head(hidden[start:start+32]).float().log_softmax(-1)
            target = torch.tensor(item['answer_ids'][start:start+32], device=model.device)
            entropy.extend((-(lp.exp()*lp).sum(-1)).cpu().tolist())
            surprisal.extend((-lp.gather(-1,target[:,None])[:,0]).cpu().tolist())
    if positions is None:
        return dict(mass=np.stack(run.mass), entropy=np.array(entropy), surprisal=np.array(surprisal),
            switch_measures=np.stack(run.switches) if run.switches else np.array([]))
    return dict(stats=np.stack(run.stats), attention=np.stack(run.attention),
        previous_attention=np.stack(run.previous_attention),
        key_projection=np.stack(run.key_projection), states=np.stack(run.states),
        pre_states=np.stack(run.pre_states), attn_states=np.stack(run.attn_states),
        site_readouts=np.stack(run.site_readouts), mlp_stats=np.stack(run.mlp_stats),
        reconstruction=np.array(run.errors), positions=np.array(positions))
