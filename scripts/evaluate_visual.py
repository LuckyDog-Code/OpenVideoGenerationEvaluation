"""Same evaluators for native and standardized views; no new lossy compression."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ap=argparse.ArgumentParser()
ap.add_argument('--view',choices=['native','standardized'],default='native')
ap.add_argument('--config',type=Path,required=True)
ap.add_argument('--run-dir',type=Path,required=True)
ap.add_argument('--vbench-dir',type=Path,required=True)
ap.add_argument('--cache-dir',type=Path,required=True)
ap.add_argument('--prompts',type=Path,default=Path(__file__).resolve().parents[1]/'data/prompts_30.jsonl')
ap.add_argument('--allow-downloads',action='store_true')
args=ap.parse_args()
cfg=json.loads(args.config.read_text(encoding='utf8'))
run=args.run_dir.resolve();out=run/'evaluation'/args.view;out.mkdir(parents=True,exist_ok=True)
infra=args.cache_dir.resolve();src=args.vbench_dir.resolve()
os.environ['VBENCH_CACHE_DIR']=str(infra/'vbench')
os.environ['HF_HOME']=str(infra/'huggingface')
os.environ['TORCH_HOME']=str(infra/'torch')
sys.path.insert(0,str(src))
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from transformers import CLIPModel,CLIPProcessor
from vbench import VBench
assert torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name(0)
torch.set_num_threads(4)
rev='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
vb_rev=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
assert vb_rev=='fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
assert not subprocess.check_output(['git','-C',str(src),'diff','HEAD','--name-only'],text=True).strip()
dims=['subject_consistency','motion_smoothness','dynamic_degree','imaging_quality','aesthetic_quality']
cache=str(infra/'huggingface')
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',revision=rev,cache_dir=cache,local_files_only=not args.allow_downloads).cuda().eval()
proc=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',revision=rev,cache_dir=cache,local_files_only=not args.allow_downloads,use_fast=False)
spec={'view':args.view,'clip_revision':rev,'vbench_revision':vb_rev,'dimensions':dims,
      'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','imageio','numpy']},
      'standardized':{'seconds':5,'fps':16,'width':832,'height':480,'spatial':'letterbox_lanczos4','temporal':'nearest_timestamp_half_up'},
      'source':'decoded imageio libx264 quality=8 video','lossless_proxy':'libx264rgb crf0 rgb24 verified exact RGB'}
(out/'spec.json').write_text(json.dumps(spec,indent=2))
deadline=time.monotonic()+5.4*3600
for sample in [json.loads(x) for x in args.prompts.read_text(encoding='utf8').splitlines()]:
    name=sample['prompt_id']+'_'+str(cfg['seed']); record=run/(name+'.json'); target=out/(name+'.json')
    if not record.exists() or target.exists(): continue
    if time.monotonic()>deadline: break
    r=json.loads(record.read_text()); r['video_path']=str(run/(name+'.mp4')); pixels=[]
    with imageio.get_reader(r['video_path']) as reader:
        for frame in reader: pixels.append(frame)
    mapping=list(range(len(pixels)))
    pad=None
    if args.view=='standardized':
        assert len(pixels)/cfg['fps']>=5
        mapping=[int(np.floor(k*cfg['fps']/16+0.5)) for k in range(80)]
        transformed=[]
        h,w=pixels[0].shape[:2]; scale=min(832/w,480/h); nw,nh=round(w*scale),round(h*scale)
        left,top=(832-nw)//2,(480-nh)//2; pad=dict(left=left,top=top,width=nw,height=nh)
        for i in mapping:
            canvas=np.zeros((480,832,3),dtype=np.uint8)
            canvas[top:top+nh,left:left+nw]=cv2.resize(pixels[i],(nw,nh),interpolation=cv2.INTER_LANCZOS4)
            transformed.append(canvas)
        pixels=transformed
    caption=sample['evaluation_prompt']; tokens=proc.tokenizer(caption,truncation=False)['input_ids']
    assert len(tokens)<=77,'CLIP caption overflow'
    idx=np.linspace(0,len(pixels)-1,16).round().astype(int).tolist()
    with torch.inference_mode():
        text=proc(text=[caption],padding=True,return_tensors='pt')
        te=model.get_text_features(**{k:v.cuda() for k,v in text.items()}); te=torch.nn.functional.normalize(te.float(),dim=-1)
        cos=[]
        for start in range(0,16,4):
            batch=proc(images=[pixels[i] for i in idx[start:start+4]],return_tensors='pt')['pixel_values'].cuda()
            em=torch.nn.functional.normalize(model.get_image_features(pixel_values=batch).float(),dim=-1)
            cos.extend((em@te[0]).cpu().tolist())
    work=out/name; work.mkdir(exist_ok=True)
    if args.view=='standardized':
        path=work/'view.mp4'
        with imageio.get_writer(str(path),fps=16,codec='libx264rgb',quality=None,pixelformat='rgb24',macro_block_size=1,
                                ffmpeg_params=['-crf','0','-preset','ultrafast']) as writer:
            for frame in pixels: writer.append_data(frame)
        with imageio.get_reader(str(path)) as reader:
            for i,decoded in enumerate(reader): assert np.array_equal(decoded,pixels[i]),'Lossless proxy RGB mismatch'
    else: path=Path(r['video_path'])
    ev=VBench(torch.device('cuda'),str(src/'vbench/VBench_full_info.json'),str(work/'raw'))
    ev.evaluate(videos_path=str(path),name='video',prompt_list=[sample['prompt']],dimension_list=dims,mode='custom_input')
    raw=json.loads((work/'raw/video_eval_results.json').read_text())
    result={'prompt_id':sample['prompt_id'],'job_id':os.environ.get('SLURM_JOB_ID'),'spec':spec,
            'video_sha256':r['video_sha256'],'frame_mapping':mapping,'letterbox':pad,'clip_indices':idx,
            'clip_raw_cosines':cos,'caption':caption,'scores':{d:float(raw[d][0]) for d in dims}}
    result['scores']['clip_x100']=100*max(0,float(np.mean(cos)))
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
    if args.view=='standardized': path.unlink() # derived lossless proxy, reproducible from retained source + mapping
    print('EVALUATED',name,args.view,result['scores'],flush=True)
if all((out/(s['prompt_id']+'_'+str(cfg['seed'])+'.json')).exists() for s in [json.loads(x) for x in args.prompts.read_text(encoding='utf8').splitlines()]):
    (out/'EVALUATION_DONE').write_text(os.environ.get('SLURM_JOB_ID','local'))
