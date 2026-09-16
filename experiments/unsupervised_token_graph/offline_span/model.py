"""层序 attention 图编码 + 证据/片段配对；训练端不是原 LLM 的消息重演。"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


class EvidenceSpanScorer(nn.Module):
    """token embedding 提供词项；可选真实hidden提供语义。头标签只在边消息里使用。"""

    def __init__(self, vocabulary, channels, node_size=0, hidden_size=32, relation='real'):
        super().__init__()
        self.vocabulary = np.asarray(vocabulary, dtype=np.int64)
        self.channels = [tuple(channel) for channel in channels]
        self.channel_lookup = {channel: index for index, channel in enumerate(self.channels)}
        self.relation = relation
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(len(vocabulary) + 1, hidden_size)
        self.side_embedding = nn.Embedding(2, hidden_size)
        self.head_embedding = nn.Embedding(len(channels), hidden_size)
        self.hidden_projection = nn.Linear(node_size, hidden_size, bias=False) if node_size else None
        self.entropy_projection = nn.Linear(2, hidden_size, bias=False)
        self.message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.update = nn.LayerNorm(hidden_size)
        self.response_projection = nn.Linear(3 * hidden_size, hidden_size)
        self.comparison = nn.Sequential(nn.Linear(4 * hidden_size, 2 * hidden_size), nn.GELU(),
                                        nn.Linear(2 * hidden_size, 1))

    @property
    def device(self):
        return self.token_embedding.weight.device

    def encode_graph(self, graph):
        """同层全部head先分别形成消息，再写入下一边界；不能在同层递归走多跳。"""
        token_ids = graph.sample.token_ids
        lookup = np.searchsorted(self.vocabulary, token_ids)
        valid = lookup < len(self.vocabulary)
        valid[valid] &= self.vocabulary[lookup[valid]] == token_ids[valid]
        encoded_ids = np.where(valid, lookup + 1, 0)
        ids = torch.as_tensor(encoded_ids, device=self.device)
        side = torch.as_tensor(np.arange(len(token_ids)) >= graph.sample.prompt_length,
                               device=self.device, dtype=torch.long)
        original = self.token_embedding(ids) + self.side_embedding(side)
        if self.hidden_projection is not None:
            features = torch.as_tensor(graph.node_features, device=self.device)
            original = original + self.hidden_projection(functional.normalize(features, dim=-1))
        # 预测熵属于q=P+t-1；缺失时显式availability=0，不伪造测量。
        uncertainty = np.zeros((len(token_ids), 2), np.float32)
        valid_entropy = np.isfinite(graph.entropy)
        queries = graph.sample.prompt_length - 1 + np.flatnonzero(valid_entropy)
        uncertainty[queries, 0] = np.log1p(graph.entropy[valid_entropy])
        uncertainty[queries, 1] = 1
        original = original + self.entropy_projection(torch.as_tensor(uncertainty, device=self.device))
        state = original

        channel_ids = torch.as_tensor([self.channel_lookup[tuple(channel)] for channel in graph.channels],
                                      device=self.device, dtype=torch.long)
        if self.relation != 'none':
            for layer in np.unique(graph.channels[:, 0]):
                selected = graph.channels[graph.edges[:, 0], 0] == layer
                edge = graph.edges[selected]
                if not len(edge):
                    continue
                source_ids = edge[:, 1].copy()
                if self.relation == 'permuted':
                    # 在CPU改检测端的配对，避免为每个query启动GPU小算子。
                    for query in np.unique(edge[:, 2]):
                        for is_prompt in (True, False):
                            mask = (edge[:, 2] == query) & ((edge[:, 1] < graph.sample.prompt_length) == is_prompt)
                            source_ids[mask] = source_ids[mask][::-1]
                source = torch.as_tensor(source_ids, device=self.device, dtype=torch.long)
                target = torch.as_tensor(edge[:, 2].copy(), device=self.device, dtype=torch.long)
                channel = torch.as_tensor(edge[:, 0].copy(), device=self.device, dtype=torch.long)
                weights = torch.as_tensor(graph.weights[selected], device=self.device)
                gate = torch.sigmoid(self.query_gate(state[target]) + self.head_embedding(channel_ids[channel]))
                messages = self.message(state[source]) * gate * weights[:, None]
                aggregate = torch.zeros_like(state).index_add(0, target, messages)
                head_count = int(np.sum(graph.channels[:, 0] == layer))
                state = self.update(state + aggregate / np.sqrt(head_count))
        return original, state

    def pool(self, states, node_lists):
        """只对明确视图内的节点汇总；空视图是缺少邻域，不是伪造hidden观测。"""
        lengths = [len(nodes) for nodes in node_lists]
        flattened = np.concatenate(node_lists) if sum(lengths) else np.empty(0, int)
        nodes = torch.as_tensor(flattened, device=self.device, dtype=torch.long)
        offsets = torch.as_tensor(np.r_[0, np.cumsum(lengths)], device=self.device, dtype=torch.long)
        return functional.embedding_bag(nodes, states, offsets, mode='mean', include_last_offset=True)

    def encode_evidence(self, encoded, spans):
        return self.pool(encoded[0], [span.evidence_nodes for span in spans])

    def encode_reuse(self, encoded, spans):
        state = encoded[1]
        content = self.pool(state, [span.response_nodes for span in spans])
        selection = self.pool(state, [span.selector_nodes for span in spans])
        later = self.pool(state, [span.later_nodes for span in spans])
        return self.response_projection(torch.cat((content, selection, later), dim=-1))

    def score_encoded(self, encoded, spans):
        evidence = self.encode_evidence(encoded, spans)
        response = self.encode_reuse(encoded, spans)
        joint = torch.cat((evidence, response, evidence * response, torch.abs(evidence - response)), dim=-1)
        return self.comparison(joint).squeeze(-1)

    def forward(self, graph, spans):
        return self.score_encoded(self.encode_graph(graph), spans)
