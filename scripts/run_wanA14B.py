"""Measured wanA14B server; one warmup, pilot, then resumable formal samples."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ap=argparse.ArgumentParser(description='Measured wanA14B generation on one L40S (Linux).')
ap.add_argument('--pilot',action='store_true')
ap.add_argument('--max-samples',type=int,default=30)
ap.add_argument('--comfy-dir',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,default=Path('runs'))
ap.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/wanA14B.json')
ap.add_argument('--prompts',type=Path,default=Path(__file__).resolve().parents[1]/'data/prompts_30.jsonl')
ap.add_argument('--pilots',type=Path,default=Path(__file__).resolve().parents[1]/'data/pilots.jsonl')
args=ap.parse_args()
cfg=json.loads(args.config.read_text(encoding='utf8'))
rows=[json.loads(x) for x in args.prompts.read_text(encoding='utf8').splitlines() if x.strip()]
out=(args.output_dir/cfg['config_id']).resolve(); out.mkdir(parents=True,exist_ok=True)
job=os.environ.get('SLURM_JOB_ID','local'); comfy=args.comfy_dir.resolve()
with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
base=f'http://127.0.0.1:{port}'; stop=threading.Event(); boot=time.perf_counter()
def request(path,data=None):
    req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)
def node(kind,**inputs): return {'class_type':kind,'inputs':inputs}
def resources():
    import psutil
    with (out/('cpu_'+job+'.jsonl')).open('w') as f:
        while not stop.wait(0.2):
            try:
                procs=[psutil.Process(server.pid)]; procs+=procs[0].children(recursive=True)
                rss=read=write=0
                for proc in procs:
                    try:
                        rss+=proc.memory_info().rss; io=proc.io_counters(); read+=io.read_bytes; write+=io.write_bytes
                    except psutil.Error: pass
                f.write(json.dumps(dict(monotonic_s=time.perf_counter(),rss_bytes=rss,read_bytes=read,write_bytes=write))+'\n'); f.flush()
            except psutil.Error: break
def generate(sample,name):
    wf={'1':node('UNETLoader',unet_name=cfg['unets'][0],weight_dtype='default'),
        '2':node('UNETLoader',unet_name=cfg['unets'][1],weight_dtype='default'),
        '3':node('CLIPLoader',clip_name=cfg['text_encoder'],type='wan',device='default'),
        '4':node('VAELoader',vae_name=cfg['vae']),
        '7':node('ModelSamplingSD3',model=['1',0],shift=cfg['shift']),
        '8':node('ModelSamplingSD3',model=['2',0],shift=cfg['shift']),
        '20':node('MLMeasuredA14B',high=['7',0],low=['8',0],clip=['3',0],vae=['4',0],
                   payload=json.dumps(dict(config=cfg,sample=sample,name=name,output_dir=str(out))))}
    (out/(name+'.workflow.json')).write_text(json.dumps(wf,indent=2))
    t=time.perf_counter()
    try: response=request('/prompt',{'prompt':wf,'client_id':'ml_h3'})
    except urllib.error.HTTPError as e:
        (out/(name+'.validation_error.json')).write_bytes(e.read()); raise
    pid=response['prompt_id']
    while time.perf_counter()-t<10000:
        if server.poll() is not None: raise RuntimeError('ComfyUI exited')
        h=request('/history/'+pid)
        if pid in h:
            (out/(name+'.history.json')).write_text(json.dumps(h[pid],indent=2))
            if h[pid]['status']['status_str']!='success': raise RuntimeError('Generation failed: '+name)
            r=json.loads((out/(name+'.json')).read_text())
            assert (r['actual_frames'],r['actual_width'],r['actual_height'])==(cfg['frames'],cfg['width'],cfg['height'])
            assert r['actual_nfe']>0 and r['torch_peak_allocated_bytes']>0
            (out/(name+'.request.json')).write_text(json.dumps({'job_id':job,'prompt_id':pid,'submit_to_observed_completion_s':time.perf_counter()-t,'not_primary_latency':True},indent=2))
            return r
        time.sleep(2)
    raise TimeoutError(name)

with (out/('comfy_'+job+'.log')).open('w') as log,(out/('gpu_'+job+'.csv')).open('w') as gpu_log:
    server=subprocess.Popen([sys.executable,'-u','main.py','--listen','127.0.0.1','--port',str(port),
        '--preview-method','none','--disable-auto-launch','--fast-disk','--disable-pinned-memory'],
        cwd=comfy,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                              '--format=csv','-lms','200'],stdout=gpu_log,stderr=subprocess.STDOUT)
    threading.Thread(target=resources,daemon=True).start()
    try:
        for _ in range(120):
            if server.poll() is not None: raise RuntimeError('Server startup failed')
            try: info=request('/object_info')
            except Exception: time.sleep(2); continue
            if 'MLMeasuredA14B' not in info: raise RuntimeError('Measurement node import failed; see server log')
            break
        else: raise TimeoutError('Server startup')
        (out/('object_info_'+job+'.json')).write_text(json.dumps(info))
        (out/('startup_'+job+'.json')).write_text(json.dumps({'startup_s':time.perf_counter()-boot}))
        warm={'prompt_id':'warmup_fixed','prompt':'A small wooden fishing boat floats on gentle turquoise waves near a sandy beach at sunrise. The camera slowly moves forward.'}
        generate(warm,'warmup_'+job)
        if args.pilot:
            sample=dict(prompt_id='pilot_camera',prompt='The camera slowly orbits clockwise around a stationary yellow bicycle beside a brick wall, with smooth parallax and consistent geometry.')
            r=generate(sample,'pilot_'+job)
            (out/'PILOT_PASSED.json').write_text(json.dumps({'job_id':job,'config_id':cfg['config_id'],'generation_wall_s':r['generation_wall_s']},indent=2))
        else:
            assert (out/'PILOT_PASSED.json').exists()
            n=0
            for sample in rows:
                name=sample['prompt_id']+'_'+str(cfg['seed'])
                if (out/(name+'.json')).exists(): continue
                if n>=args.max_samples or time.perf_counter()-boot>4.8*3600: break
                generate(sample,name); n+=1
            if all((out/(r['prompt_id']+'_'+str(cfg['seed'])+'.json')).exists() for r in rows):
                (out/'GENERATION_DONE').write_text(job)
    finally:
        stop.set(); monitor.terminate()
        if server.poll() is None: os.killpg(server.pid,signal.SIGTERM)
        try: server.wait(timeout=30)
        except subprocess.TimeoutExpired: os.killpg(server.pid,signal.SIGKILL)
