"""ComfyUI output node: loaders outside timer; complete A14B generation inside."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import imageio.v2 as imageio
import nodes

def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

class MLMeasuredA14B:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'high': ('MODEL',), 'low': ('MODEL',), 'clip': ('CLIP',),
                             'vae': ('VAE',), 'payload': ('STRING', {'multiline': True})}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = 'ML experiment'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, high, low, clip, vae, payload):
        p = json.loads(payload)
        cfg, sample = p['config'], p['sample']
        root = Path(p['output_dir'])
        root.mkdir(parents=True, exist_ok=True)
        name = p['name']
        target = root / (name + '.json')
        assert torch.cuda.device_count() == 1
        assert 'L40S' in torch.cuda.get_device_name(0), 'L40S required'
        stages, counts, handles = {}, {}, []
        dtype_counts = {}
        for label, model in [('high', high), ('low', low)]:
            net = model.model.diffusion_model
            dtype_counts[label] = dict(collections.Counter(str(v.dtype) for v in net.parameters()))
            counts[label] = []
            original = net.forward
            def wrapper(*a, _original=original, _label=label, **kw):
                x = a[0] if a else kw.get('x')
                counts[_label].append(list(x.shape) if hasattr(x, 'shape') else None)
                return _original(*a, **kw)
            net.forward = wrapper
            handles.append((net, original))
        def stage(label, fn):
            t = time.perf_counter()
            value = fn()
            stages[label] = time.perf_counter() - t
            return value
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        record = dict(model='Wan2.2-T2V-A14B', sample=sample, config=cfg,
                      job_id=os.environ.get('SLURM_JOB_ID'), hostname=os.uname().nodename,
                      gpu=torch.cuda.get_device_name(0), dtype_parameter_tensor_counts=dtype_counts,
                      name=name, status='started', baseline_allocated_bytes=baseline,
                      generation_start_monotonic_s=t0,generation_start_unix_s=time.time())
        try:
            positive = stage('positive_text', lambda: nodes.CLIPTextEncode().encode(clip, sample['prompt'])[0])
            negative = stage('negative_text', lambda: nodes.CLIPTextEncode().encode(clip, cfg['negative_prompt'])[0])
            latent = {'samples': torch.zeros((1,16,(cfg['frames']-1)//4+1,cfg['height']//8,cfg['width']//8))}
            record['latent_shape'] = list(latent['samples'].shape)
            record['video_token_count'] = int(latent['samples'].shape[2] * latent['samples'].shape[3]//2 * (latent['samples'].shape[4]//2))
            def sample_stage(model, latent, begin, end, noise, leftover):
                return nodes.KSamplerAdvanced().sample(model, noise, cfg['seed'], cfg['steps'], cfg['cfg'],
                    cfg['sampler'], cfg['scheduler'], positive, negative, latent, begin, end, leftover)[0]
            latent = stage('high_denoise', lambda: sample_stage(high,latent,0,cfg['expert_split_step'],'enable','enable'))
            latent = stage('low_denoise', lambda: sample_stage(low,latent,cfg['expert_split_step'],cfg['steps'],'disable','disable'))
            images = stage('video_decode', lambda: nodes.VAEDecodeTiled().decode(vae,latent,**cfg['tiled_decode'])[0])
            torch.cuda.synchronize()
            end=time.perf_counter()
            record.update(generation_wall_s=end-t0,generation_end_monotonic_s=end,
                          torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          stage_wall_s=stages, denoiser_call_shapes=counts,
                          actual_frames=int(images.shape[0]), actual_height=int(images.shape[1]),
                          actual_width=int(images.shape[2]), status='generated')
            atomic(target.with_suffix('.generated.json'), record)
            t_encode = time.perf_counter()
            video = root / (name + '.mp4')
            with imageio.get_writer(str(video), fps=cfg['fps'], codec='libx264', quality=8,
                                    macro_block_size=1, pixelformat='yuv420p',
                                    ffmpeg_params=['-preset','medium']) as writer:
                for frame in images:
                    rgb = (frame.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                    writer.append_data(rgb)
            reader=imageio.get_reader(str(video)); metadata=reader.get_meta_data(); decoded=sum(1 for _ in reader); reader.close()
            assert decoded==record['actual_frames']
            record.update(encode_write_s=time.perf_counter()-t_encode, video_path=str(video),
                          actual_duration_s=record['actual_frames']/cfg['fps'],decoded_frame_count=decoded,
                          video_metadata=metadata,actual_nfe=sum(map(len,counts.values())),
                          encoder=dict(codec='libx264',quality=8,preset='medium',pixel_format='yuv420p',macro_block_size=1),
                          container_bitrate_bps=video.stat().st_size*8/(record['actual_frames']/cfg['fps']),
                          file_bytes=video.stat().st_size,
                          video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),status='success')
            atomic(target, record)
            print('ML_RESULT', name, record['generation_wall_s'], record['torch_peak_allocated_bytes'],flush=True)
        except Exception as exc:
            record.update(status='failed', error=repr(exc), stage_wall_s=stages, denoiser_call_shapes=counts)
            atomic(target.with_suffix('.failed.json'),record)
            raise
        finally:
            for net, original in handles:
                net.forward = original
        return {'ui': {'text': [str(target)]}}

NODE_CLASS_MAPPINGS = {'MLMeasuredA14B': MLMeasuredA14B}
