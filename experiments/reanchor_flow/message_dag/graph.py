"""Two sweeps over a source-resolved native-reference computation DAG.

Forward: x_v[c] = boundary_v[c] + sum_e B_e x_u[c].
Reverse: lambda_u = sum_e B_e.T lambda_v, for ONE fixed output target.
Edge: C_e[c,t] = lambda_v.T B_e x_u[c]. Signed flow balances at internal
nodes. Absolute flow is descriptive throughput, not a conserved probability.
"""
from pathlib import Path

import numpy as np
import torch

from ..message_lineage import readout_directions
from .cache import source_partition
from .operators import LayerOperator

SCHEMA = 1


class Tape:
    """Uncompressed temporary source states, bounded GPU residency by groups."""
    def __init__(self, cache, directory):
        self.partition = source_partition(cache.trace)
        self.groups = len(self.partition["names"])
        self.shape = (cache.layers, self.groups, cache.rows, cache.weights.config["hidden_size"])
        directory = Path(directory)
        self.input = np.lib.format.open_memmap(directory/"input.npy", mode="w+", dtype="float32", shape=self.shape)
        self.post = np.lib.format.open_memmap(directory/"post.npy", mode="w+", dtype="float32", shape=self.shape)
        self.error = np.zeros((cache.layers, 2, cache.rows, self.shape[-1]), np.float32)

    def close(self):
        for array in (self.input, self.post):
            array.flush()
            array._mmap.close()


def blocks(n, size):
    for begin in range(0, n, size):
        yield begin, min(n, begin+size)


@torch.inference_mode()
def prepare(cache, tape, *, source_chunk=2, query_chunk=8, rule="symmetric", progress=None):
    """Propagate every source unit once; targets reuse this temporary tape."""
    partition = tape.partition
    x = cache.states["residual_0"]
    parts = partition["initial"][..., None].astype(np.float32) * x[None]
    errors = []
    for layer in range(cache.layers):
        if progress: progress(f"source propagation: L{layer}/{cache.layers-1}")
        op = LayerOperator(cache, layer, rule, query_chunk)
        tape.input[layer] = parts
        post_sum = torch.zeros_like(op.x)
        for a, b in blocks(tape.groups, source_chunk):
            p = op.tensor(parts[a:b])
            post = p + op.attention(p, op.tensor(partition["boundary"][a:b]))
            tape.post[layer, a:b] = post.cpu().numpy()
            post_sum += post.sum(0)
        error_post = op.post - post_sum
        tape.error[layer, 0] = error_post.cpu().numpy()
        tape.post[layer, partition["rounding"]] += tape.error[layer, 0]
        next_sum = torch.zeros_like(op.x)
        mlp_sum = torch.zeros_like(op.x)
        for a, b in blocks(tape.groups, source_chunk):
            post = op.tensor(np.array(tape.post[layer, a:b]))
            mlp = op.mlp(post)
            nxt = post + mlp
            parts[a:b] = nxt.cpu().numpy()
            next_sum += nxt.sum(0)
            mlp_sum += mlp.sum(0)
        error_next = op.next - next_sum
        tape.error[layer, 1] = error_next.cpu().numpy()
        parts[partition["rounding"]] += tape.error[layer, 1]
        # Validate module writes, separately from BF16 residual-add rounding.
        # A small MLP write can legitimately be below the residual's rounding
        # scale; dividing addition error by that write gives a false mismatch.
        native_attention = op.tensor(cache.states[f"attention_{layer}"])
        native_mlp = op.tensor(cache.states[f"mlp_{layer}"])
        relative = [float((a-b).norm()/b.norm().clamp_min(1e-8)) for a,b in
                    ((post_sum-op.x,native_attention),(mlp_sum,native_mlp))]
        if not np.isfinite(relative).all() or max(relative) > .05:
            raise ValueError(f"L{layer}: native replay error {relative}; check checkpoint/capture compatibility")
        errors.append(relative)
        del op
    tape.final = parts
    tape.replay_error = np.asarray(errors)
    tape.direction, tape.runner = readout_directions(cache.trace, cache.states, cache.weights, query_chunk)
    tape.final_credit = np.einsum("grd,rd->gr", parts, tape.direction.cpu().numpy())
    return tape


def _select_edges(scores, credits, positions, query_positions, layer, budget, records, attention, value_norm):
    """Display only. All edges already enter propagation, balance and metrics."""
    flat = scores.flatten()
    k = min(budget, flat.numel())
    if not k: return
    values, indices = flat.topk(k)
    indices, values = indices[values>0], values[values>0]
    if not len(indices): return
    h, q, s = scores.shape
    hi, qi, si = indices//(q*s), (indices//s)%q, indices%s
    coordinates=torch.stack((hi,qi,si),-1).cpu().numpy()
    native=torch.stack((attention[hi,qi,si],value_norm[hi,si]),-1).cpu().numpy()
    if isinstance(credits,tuple):
        boundary,groups,group_count=credits
        selected=np.zeros((len(indices),group_count),np.float32)
        selected[np.arange(len(indices)),groups[coordinates[:,2]]]=boundary[hi,qi,si].cpu().numpy()
    else:
        selected=credits[:,hi,qi,si].T.cpu().numpy()
    for strength,(hi,qi,si),value,(a,vnorm) in zip(values.cpu().tolist(),coordinates,selected,native):
        records.append((strength, [layer,int(hi),int(positions[si]),int(query_positions[qi])],value,float(a),float(vnorm)))
    records.sort(key=lambda x: -x[0])
    del records[budget:]


def _target_steps(cache, tape, target, *, source_chunk=2, edge_budget=24, progress=None):
    """Exact (within allocation arithmetic) all-path contribution for token t."""
    trace, partition = cache.trace, tape.partition
    rows = trace["row_position"]
    where = np.flatnonzero(rows == target-1)
    if not len(where) or target >= len(trace["token_ids"]) or trace["special_mask"][[target-1,target]].any():
        raise ValueError("target must have an observed ordinary predictor and token")
    target_row = int(where[0])
    g, l, r, h = tape.groups, cache.layers, cache.rows, cache.heads
    part = lambda shape: np.zeros(shape, np.float32)
    result = {"schema": np.array(SCHEMA), "target": np.array(target), "query": np.array(target-1),
              "runner": np.array(tape.runner[target_row]), "source_names": partition["names"],
              "source_kinds": partition["kinds"], "row_position": rows,
              "source_output": tape.final_credit[:, target_row], "replay_error": tape.replay_error,
              "node_input": part((g,l+1,r)), "node_post": part((g,l,r)),
              "node_out_absolute": part((g,l,r)), "mlp_credit": part((g,l,r)),
              "head_relay": part((g,l,h)), "head_relay_absolute": part((g,l,h)),
              "head_boundary": part((g,l,h)), "carrier_absolute": part((g,l,r)),
              "eligible_absolute_by_source": part((g,)),
              "local_relay_absolute": part((g,l,h)), "root_credit": part((g,)),
              "boundary_at_target": part((g,)), "root_position_credit": part((len(trace['token_ids']),)),
              "balance_error": part((l,2)), "labels_used": np.array(False)}
    adjoint = torch.zeros_like(tape.direction)
    adjoint[target_row] = tape.direction[target_row]
    result["node_input"][:, -1, target_row] = result["source_output"]
    edge_records, eligible_gross = [], 0.0
    special = trace["special_mask"]
    ordinary_carrier = (rows >= int(trace["response_start"])) & ~special[rows]
    window = cache.trace.get("settings", np.array('{}'))
    import json
    window = json.loads(str(window)).get("local_window", 10)
    for layer in reversed(range(l)):
        if progress: progress(f"target {target}: adjoint and edges L{layer}/{l-1}")
        op = yield layer
        post_adjoint = adjoint + op.mlp_adjoint(adjoint)
        input_adjoint = post_adjoint + op.attention_adjoint(post_adjoint[None],stop=target_row+1)[0]
        projected = op.output_direction(post_adjoint[None])[0]
        incoming, outgoing = op.tensor(part((g,r))), op.tensor(part((g,r)))
        in_absolute, out_absolute = torch.zeros_like(incoming), torch.zeros_like(outgoing)
        build_values = op.source_values is None
        if build_values:
            op.source_values = torch.empty((g,op.kv,r,op.hd), device=op.device)
        pv = op.source_values
        for a, b in blocks(g, source_chunk):
            p = op.tensor(np.array(tape.input[layer,a:b]))
            post = op.tensor(np.array(tape.post[layer,a:b]))
            if build_values: pv[a:b] = op.values(p)
            result["node_input"][a:b,layer] = (p*input_adjoint).sum(-1).cpu().numpy()
            result["node_post"][a:b,layer] = (post*post_adjoint).sum(-1).cpu().numpy()
            result["mlp_credit"][a:b,layer] = (post*(post_adjoint-adjoint)).sum(-1).cpu().numpy()
            residual = (p*post_adjoint).sum(-1)
            incoming[a:b] += residual
            outgoing[a:b] += residual
            in_absolute[a:b] += residual.abs()
            out_absolute[a:b] += residual.abs()
        native_v = op.value.repeat_interleave(h//op.kv, dim=0)
        value_norm=native_v.norm(dim=-1)
        start=int(trace['response_start'])
        group_index=torch.as_tensor(partition['token_group'][:start],device=op.device)
        boundary_total=torch.zeros(g,device=op.device)
        boundary_head=torch.zeros((g,h),device=op.device)
        boundary_direct=torch.zeros(g,device=op.device)
        boundary_eligible=torch.zeros(g,device=op.device)
        boundary_positions=torch.zeros(start,device=op.device)
        layer_edges = []
        for begin, end, attention in op.rows_attention(stop=target_row+1):
            q = rows[begin:end]
            projected_q = projected[:, begin:end]
            response = torch.empty((g,h,end-begin,r), device=op.device)
            for a,b in blocks(g, source_chunk):
                values = pv[a:b].repeat_interleave(h//op.kv, dim=1)
                response[a:b] = torch.einsum("hqd,ghsd->ghqs", projected_q, values) * attention[None,...,op.rows]
            incoming[:,begin:end] += response.sum((1,3))
            outgoing += response.sum((1,2))
            in_absolute[:,begin:end] += response.abs().sum((1,3))
            out_absolute += response.abs().sum((1,2))
            # Special nodes and self edges stay in the computation and balance.
            # Ordinary strict-past edges define the displayed/compared relays.
            relay = (rows[None] < q[:,None]) & ordinary_carrier[None] & ~special[q,None]
            local = relay & (q[:,None]-rows[None] <= window)
            c = response * op.tensor(relay)[None,None]
            result["head_relay"][:,layer] += c.sum((2,3)).cpu().numpy()
            result["head_relay_absolute"][:,layer] += c.abs().sum((2,3)).cpu().numpy()
            result["local_relay_absolute"][:,layer] += (response.abs()*op.tensor(local)[None,None]).sum((2,3)).cpu().numpy()
            result["carrier_absolute"][:,layer] += c.abs().sum((1,2)).cpu().numpy()
            gross = c.abs().sum(0)
            eligible_gross += float(gross.sum())
            result["eligible_absolute_by_source"] += c.abs().sum((1,2,3)).cpu().numpy()
            _select_edges(gross,c,rows,q,layer,edge_budget,layer_edges,attention[...,op.rows],value_norm[:,op.rows])
            boundary = (torch.einsum("hqd,hsd->hqs", projected_q, native_v[:,:start])
                        * attention[:,:,:start])
            # Scatter into disjoint source groups: no G x H x Q x prompt
            # tensor and no device synchronization for every source unit.
            def grouped(values):
                sums=torch.zeros((h,end-begin,g),device=op.device)
                sums.scatter_add_(2,group_index[None,None].expand_as(values),values)
                return sums.permute(2,0,1)
            net,absolute=grouped(boundary),grouped(boundary.abs())
            incoming[:,begin:end] += net.sum(1)
            in_absolute[:,begin:end] += absolute.sum(1)
            boundary_head += net[:,:,~special[q]].sum(2)
            boundary_total += net.sum((1,2))
            if begin<=target_row<end: boundary_direct += net[:,:,target_row-begin].sum(1)
            boundary_positions += boundary.sum((0,1))
            eligible = ~special[:start][None] & ~special[q,None] & (np.arange(start)[None] < q[:,None])
            gross = boundary.abs() * op.tensor(eligible)[None]
            boundary_eligible += grouped(gross).sum((1,2))
            eligible_gross += float(gross.sum())
            _select_edges(gross,(boundary,partition['token_group'][:start],g),np.arange(start),q,layer,
                          edge_budget,layer_edges,attention[:,:,:start],value_norm[:,:start])
        result['root_credit'] += boundary_total.cpu().numpy()
        result['head_boundary'][:,layer] = boundary_head.cpu().numpy()
        result['boundary_at_target'] += boundary_direct.cpu().numpy()
        result['root_position_credit'][:start] += boundary_positions.cpu().numpy()
        result['eligible_absolute_by_source'] += boundary_eligible.cpu().numpy()
        round_id = partition["rounding"]
        injection_post = (op.tensor(tape.error[layer,0])*post_adjoint).sum(-1)
        injection_next = (op.tensor(tape.error[layer,1])*adjoint).sum(-1)
        incoming[round_id] += injection_post
        result["root_credit"][round_id] += float(injection_post.sum()+injection_next.sum())
        result["node_out_absolute"][:,layer] = out_absolute.cpu().numpy()
        for j,(observed, expected) in enumerate(((outgoing, result['node_input'][:,layer]),
                                                (incoming, result['node_post'][:,layer]))):
            expected = op.tensor(expected)
            scale = torch.maximum(expected.abs().max(), observed.abs().max()).clamp_min(1e-6)
            result["balance_error"][layer,j] = float((observed-expected).abs().max()/scale)
        edge_records.extend(layer_edges)
        adjoint = input_adjoint
        del op, pv, response
    initial = np.array(tape.input[0])
    initial_credit = np.einsum("grd,rd->gr", initial, adjoint.cpu().numpy())
    result["root_credit"] += initial_credit.sum(-1)
    result["root_position_credit"][rows] += initial_credit.sum(0)
    result["root_balance_error"] = result["root_credit"]-result["source_output"]
    scale = max(1e-6, float(np.abs(result["source_output"]).max()))
    if (not np.isfinite(result['balance_error']).all() or result['balance_error'].max() > .002
            or np.max(np.abs(result['root_balance_error']))/scale > .002):
        raise ValueError("signed DAG flow does not balance; no mechanism report was saved")
    result["edge_index"] = np.array([e[1] for e in edge_records],int).reshape(-1,4)
    result["edge_source_credit"] = np.array([e[2] for e in edge_records],np.float32).reshape(-1,g)
    result['edge_attention'] = np.array([e[3] for e in edge_records],np.float32)
    result['edge_value_norm'] = np.array([e[4] for e in edge_records],np.float32)
    result["edge_display_coverage"] = np.array(sum(e[0] for e in edge_records)/eligible_gross if eligible_gross else np.nan)
    result["eligible_edge_absolute"] = np.array(eligible_gross)
    result["output_margin"] = np.array(result["source_output"].sum())
    return result


@torch.inference_mode()
def trace_targets(cache, tape, targets, *, source_chunk=2, query_chunk=8, rule='symmetric',
                  edge_budget=24, progress=None):
    """A bounded block of independent sinks shares each loaded native layer.

    Targets are never summed: each keeps its own adjoint, graph and label
    coordinate. This reduces weight reads, decompression and source WV work.
    """
    steps=[_target_steps(cache,tape,t,source_chunk=source_chunk,edge_budget=edge_budget,progress=progress)
           for t in targets]
    requests=[next(step) for step in steps]
    results=[None]*len(steps)
    for layer in reversed(range(cache.layers)):
        op=LayerOperator(cache,layer,rule,query_chunk)
        for i,step in enumerate(steps):
            if requests[i]!=layer: raise RuntimeError('DAG layer scheduler lost topological order')
            try: requests[i]=step.send(op)
            except StopIteration as done: results[i]=done.value
        del op
    return results


def trace_target(cache,tape,target,**kwargs):
    return trace_targets(cache,tape,[target],**kwargs)[0]
