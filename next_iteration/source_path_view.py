"""Lossless typed source tree for structured inputs, not a hallucination score.

Only JSON / literal dictionaries and lists are admitted. Keys, order, null,
booleans, empty containers and full text values survive a verified round trip.
No generated evidence, task/label lookup, field filtering or time conversion.
"""
import ast
import json
import math


def _pairs(items):
    out={}
    for key,value in items:
        if not isinstance(key,str) or key in out:raise ValueError('nonstring or duplicate key')
        out[key]=value
    return out


def parse_source(source):
    try:
        obj=json.loads(source,object_pairs_hook=_pairs,
            parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f'nonfinite {s}')))
    except json.JSONDecodeError:
        tree=ast.parse(source,mode='eval')
        # Reject duplicate keys before literal_eval could silently discard them.
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                keys=[ast.literal_eval(k) for k in node.keys]
                if not all(type(k) is str for k in keys) or len(set(keys))!=len(keys):
                    raise ValueError('nonstring or duplicate literal key')
        obj=ast.literal_eval(tree)
    if type(obj) is not dict:raise ValueError('structured root must be a dictionary')
    return obj


def typed_tree(obj):
    nodes=[]; edges=[]
    def visit(value,path,parent=None,slot=None):
        kind=type(value).__name__
        if kind not in {'dict','list','str','int','float','bool','NoneType'}:
            raise ValueError(f'unsupported type: {kind}')
        if kind=='float' and not math.isfinite(value):raise ValueError('nonfinite value')
        node_id=len(nodes)
        node=dict(id=node_id,path=path,kind=kind)
        nodes.append(node)
        if parent is not None:edges.append(dict(parent=parent,child=node_id,slot=slot))
        if kind in {'dict','list'}:
            items=value.items() if kind=='dict' else enumerate(value)
            for key,child in items:
                if kind=='dict' and type(key) is not str:raise ValueError('dictionary keys must be strings')
                visit(child,path+[key],node_id,key)
            node['size']=len(value)
        else:node['value']=value
        return node_id
    visit(obj,[])
    return dict(nodes=nodes,edges=edges)


def restore(tree):
    nodes=tree['nodes']; edges=tree['edges']
    if not nodes or [n['id'] for n in nodes]!=list(range(len(nodes))):raise ValueError('node IDs')
    children={i:[] for i in range(len(nodes))}
    seen=set()
    for edge in edges:
        p,c=edge['parent'],edge['child']
        if not (0<=p<c<len(nodes)) or c in seen:raise ValueError('not an ordered rooted tree')
        seen.add(c);children[p].append(edge)
    if seen!=set(range(1,len(nodes))):raise ValueError('disconnected tree')
    def visit(i,path):
        node=nodes[i]; kind=node['kind']; descendants=children[i]
        if node['path']!=path:raise ValueError('path/edge binding mismatch')
        if kind in {'dict','list'}:
            if node['size']!=len(descendants):raise ValueError('container size mismatch')
            if kind=='list':
                if [e['slot'] for e in descendants]!=list(range(node['size'])):raise ValueError('list order')
                return [visit(e['child'],path+[e['slot']]) for e in descendants]
            return _pairs([(e['slot'],visit(e['child'],path+[e['slot']])) for e in descendants])
        if descendants or type(node['value']).__name__!=kind:raise ValueError('scalar type/edges')
        if kind not in {'str','int','float','bool','NoneType'}:raise ValueError('unknown scalar')
        return node['value']
    return visit(0,[])


def prepare(source):
    obj=parse_source(source)
    graph=typed_tree(obj)
    recovered=restore(graph)
    # JSON encodes bool/null/number types distinctly and retains insertion order.
    before=json.dumps(obj,ensure_ascii=False,allow_nan=False)
    after=json.dumps(recovered,ensure_ascii=False,allow_nan=False)
    if before!=after:raise ValueError('lossy projection')
    lines=[]
    for node in graph['nodes']:
        payload={'path':node['path'],'type':node['kind']}
        payload.update({'size':node['size']} if node['kind'] in {'dict','list'} else {'value':node['value']})
        lines.append(json.dumps(payload,ensure_ascii=False,allow_nan=False))
    return dict(tree=graph,text='\n'.join(lines),roundtrip_exact_typed_values=True,
                scope='lossless parsed source representation only; not learned graph or detector result')
