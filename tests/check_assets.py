"""CPU checks for frozen prompt coverage, configs, API graphs and Python syntax."""
import ast
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    prompts=[json.loads(x) for x in (ROOT/'data/prompts_30.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    assert len(prompts)==len({p['prompt_id'] for p in prompts})==30
    assert sorted(collections.Counter(p['selection_group'] for p in prompts).values())==[5]*6
    assert all(p['seed']==20261118 and p['prompt'] and p['evaluation_prompt'] for p in prompts)
    for p in prompts:
        assert hashlib.sha256(p['prompt'].encode()).hexdigest()==p['prompt_sha256']
    for model in ['wan5b','wanA14B','h3']:
        cfg=json.loads((ROOT/'configs'/f'{model}.json').read_text())
        wf=json.loads((ROOT/'workflows'/f'{model}.api.json').read_text())
        assert cfg['gpu_count']==1 and cfg['steps']==20 and cfg['seed']==20261118
        payloads=[json.loads(n['inputs']['payload']) for n in wf.values() if 'payload' in n['inputs']]
        assert len(payloads)==1
        assert payloads[0]['config']==cfg
        assert payloads[0]['output_dir']=='runs/'+cfg['config_id']
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf8'),filename=str(path))
    print('PASS: all 30 prompts, six groups, exact text hashes, frozen workflow configs, Python syntax')


if __name__=='__main__':main()
