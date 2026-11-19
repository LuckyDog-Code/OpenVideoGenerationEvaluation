"""Run from ~/ml_project/h3_av_v1 (pinned ImageBind checkout on PYTHONPATH)."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from PIL import Image
from torchvision import transforms as T
from torch.nn.functional import normalize, pad
ap=argparse.ArgumentParser(description='ImageBind original-pair versus mismatch test and WAV checks.')
ap.add_argument('--imagebind-dir',type=Path,required=True)
ap.add_argument('--weights',type=Path,required=True)
ap.add_argument('--run-dir',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--prompts',type=Path,default=Path(__file__).resolve().parents[1]/'data/prompts_30.jsonl')
ap.add_argument('--smoke',action='store_true')
args=ap.parse_args()
import types
pkg=types.ModuleType('imagebind');pkg.__path__=[str(args.imagebind_dir.resolve()/'imagebind')]
sys.modules['imagebind']=pkg
from imagebind.models.imagebind_model import imagebind_huge
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
run=args.run_dir.resolve()
rows=[json.loads(x) for x in args.prompts.read_text(encoding='utf8').splitlines() if x.strip()]
ids = [x['prompt_id'] for x in rows]
assert len(set(ids)) == len(ids) == 30 and all(x['seed']==20261118 for x in rows)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def write_csv(name, records):
    with (out/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)

gpu = torch.cuda.get_device_name()
assert 'L40S' in gpu, gpu
torch.set_num_threads(3)
transform = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(224), T.ToTensor(),
                       T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
weights = args.weights.resolve()
model = imagebind_huge(pretrained=False)
model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
model.eval().cuda()
smoke = args.smoke
qc, vs, aus = [], [], []
for ident in ids[:1] if smoke else ids:
    video = run/f'{ident}_20261118.mp4'
    wav = video.with_suffix('.wav')
    cached = out/f'{ident}.npz'
    vh, ah = digest(video), digest(wav)
    if not cached.exists():
        y,sr = sf.read(wav, always_2d=True)
        assert len(y) and np.isfinite(y).all()
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = min(nf/fps, len(y)/sr)
        assert duration >= 2
        starts = np.linspace(0,duration-2,3)
        frame_idx = np.floor((starts+1)*fps).astype(int)
        frames, mels = [], []
        mono = ta.functional.resample(torch.tensor(y.mean(axis=1),dtype=torch.float32)[None],sr,16000)
        for start,idx in zip(starts,frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
            ok,bgr = cap.read()
            assert ok, (ident,idx)
            frames.append(transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))))
            offset = round(start*16000)
            clip = mono[:,offset:offset+32000]
            assert clip.shape[1]==32000
            mel = ta.compliance.kaldi.fbank(clip-clip.mean(), htk_compat=True, sample_frequency=16000,
                       use_energy=False,window_type='hanning',num_mel_bins=128,dither=0.,
                       frame_length=25,frame_shift=10).T
            mels.append(((pad(mel,(0,204-mel.shape[1]))+4.268)/9.138)[None])
        cap.release()
        with torch.inference_mode():
            emb = model({'vision':torch.stack(frames).cuda(),'audio':torch.stack(mels).cuda()})
            v,a = [normalize(normalize(emb[k],dim=1).mean(0),dim=0).cpu().numpy() for k in ('vision','audio')]
        win = round(sr*.02)
        blocks = y[:len(y)//win*win].reshape(-1,win,y.shape[1])
        r = dict(prompt_id=ident,video_sha256=vh,wav_sha256=ah,sample_rate=sr,
                 channels=y.shape[1],samples_per_channel=len(y),wav_subtype=sf.info(wav).subtype,
                 peak_abs=np.abs(y).max(),rms=np.sqrt((y*y).mean()),near_clip_fraction=(np.abs(y)>=.999).mean(),
                 near_silent_fraction=((blocks**2).mean(axis=(1,2))<1e-6).mean(),
                 audio_s=len(y)/sr,video_s=nf/fps,audio_minus_video_s=len(y)/sr-nf/fps,
                 frames=nf,fps=fps,window_starts_s=json.dumps(starts.tolist()),frame_indices=json.dumps(frame_idx.tolist()))
        with cached.with_suffix('.tmp').open('wb') as f:
            np.savez(f,v=v,a=a,record=json.dumps(r))
        os.replace(cached.with_suffix('.tmp'),cached)
    with np.load(cached) as f:
        r=json.loads(str(f['record']))
        assert r['video_sha256']==vh and r['wav_sha256']==ah
        qc.append(r); vs.append(f['v']); aus.append(f['a'])
    print(ident,flush=True)
if smoke:
    sys.exit(0)
write_csv('wav_qc.csv',qc)
s = np.array(vs,dtype=float) @ np.array(aus,dtype=float).T
assert s.shape==(30,30) and np.isfinite(s).all()
write_csv('similarity.csv',[dict(video_id=i,**dict(zip(ids,row))) for i,row in zip(ids,s)])
matched = s.diagonal()
wrong = (s.sum(1)-matched)/29
write_csv('paired.csv',[dict(prompt_id=i,category=r['selection_group'],matched=m,mismatched_mean=w,advantage=m-w)
                         for i,r,m,w in zip(ids,rows,matched,wrong)])
rng = np.random.default_rng(20261118)
def advantage(m):
    return float((m-(s.sum(1)-m)/29).mean())
null = np.array([advantage(s[np.arange(30),rng.permutation(30)]) for _ in range(10000)])
np.save(out/'permutation_null.npy',null)
obs = advantage(matched)
summary = dict(n=30,matched_mean=matched.mean(),mismatched_mean=wrong.mean(),mean_advantage=obs,
               one_sided_p=(1+int((null>=obs).sum()))/10001,permutations=len(null),seed=20261118,
               gpu=gpu,job=os.environ.get('SLURM_JOB_ID','local'),weights_sha256=digest(weights))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
