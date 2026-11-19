"""Instrument the stock ComfyUI H3 nodes, retaining loaders outside generation timer."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import imageio
import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
import torch
import nodes
from comfy_extras import nodes_custom_sampler as sampling
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras import nodes_audio

def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def shapes(x):
    if getattr(x,'is_nested',False): return [list(y.shape) for y in x.unbind()]
    return list(x.shape) if hasattr(x,'shape') else None

class MLMeasuredH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'clip':('CLIP',),'vae':('VAE',),
                            'audio_vae':('VAE',),'payload':('STRING',{'multiline':True})}}
    RETURN_TYPES=()
    FUNCTION='run'
    OUTPUT_NODE=True
    CATEGORY='ML experiment'
    @classmethod
    def IS_CHANGED(cls,**kwargs): return float('nan')

    def run(self,model,clip,vae,audio_vae,payload):
        p=json.loads(payload); cfg=p['config']; sample=p['sample']; name=p['name']
        root=Path(p['output_dir']); root.mkdir(parents=True,exist_ok=True)
        target=root/(name+'.json')
        assert torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name(0)
        record=dict(name=name,model=cfg['model'],sample=sample,config=cfg,
                    job_id=os.environ.get('SLURM_JOB_ID'),hostname=os.uname().nodename,
                    gpu=torch.cuda.get_device_name(0),status='started',
                    measurement_method='server_perf_counter_cuda_sync_boundaries',
                    stage_method='inclusive_node_wall_without_internal_cuda_sync')
        net=model.model.diffusion_model
        record['dtype_parameter_tensor_counts']=dict(collections.Counter(str(v.dtype) for v in net.parameters()))
        record['video_vae_dtype']=str(getattr(vae,'vae_dtype',None))
        record['audio_vae_dtype']=str(getattr(audio_vae,'vae_dtype',None))
        calls=[]; original=net.forward
        def wrapped(*a,**kw):
            x=a[0] if a else kw.get('x')
            calls.append(shapes(x))
            return original(*a,**kw)
        net.forward=wrapped
        stages={}
        def stage(label,fn):
            start=time.perf_counter(); value=fn()
            stages[label]=time.perf_counter()-start
            return value
        torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        record['baseline_allocated_bytes']=torch.cuda.memory_allocated()
        record['baseline_reserved_bytes']=torch.cuda.memory_reserved()
        start=time.perf_counter()
        record['generation_start_monotonic_s']=start
        record['generation_start_unix_s']=time.time()
        try:
            cond,latent=stage('text_encode_latent',lambda:h3.MiniMaxH3ImageToVideo.execute(
                clip,vae,sample['prompt'],cfg['width'],cfg['height'],cfg['frames']).result)
            record['latent_shapes']=shapes(latent['samples'])
            noise=sampling.RandomNoise.execute(cfg['seed'])[0]
            guider=sampling.BasicGuider.execute(model,cond)[0]
            sampler=sampling.KSamplerSelect.execute(cfg['sampler'])[0]
            sigmas=sampling.BasicScheduler.execute(model,cfg['scheduler'],cfg['steps'],1.0)[0]
            record['sigmas']=sigmas.tolist()
            sampled=stage('joint_denoise',lambda:sampling.SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent)[0])
            images=stage('video_decode',lambda:nodes.VAEDecodeTiled().decode(vae,sampled,**cfg['tiled_decode'])[0])
            audio=stage('audio_decode',lambda:nodes_audio.VAEDecodeAudio.execute(audio_vae,sampled)[0])
            torch.cuda.synchronize()
            end=time.perf_counter()
            record.update(generation_wall_s=end-start,generation_end_monotonic_s=end,
                          torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          stage_wall_s=stages,denoiser_call_shapes=calls,actual_nfe=len(calls),
                          actual_frames=int(images.shape[0]),actual_height=int(images.shape[1]),actual_width=int(images.shape[2]),
                          audio_shape=list(audio['waveform'].shape),audio_sample_rate=audio['sample_rate'],status='generated')
            atomic(target.with_suffix('.generated.json'),record)
            encode_start=time.perf_counter(); video=root/(name+'.mp4')
            with iio.get_writer(str(video),fps=cfg['fps'],codec='libx264',quality=8,macro_block_size=1,
                pixelformat='yuv420p',ffmpeg_params=['-preset','medium'],ffmpeg_log_level='info') as writer:
                for frame in images:
                    writer.append_data((frame.detach().cpu().numpy()*255).clip(0,255).astype(np.uint8))
            wav_path=root/(name+'.wav')
            wav=audio['waveform'][0].detach().cpu().numpy()
            with wave.open(str(wav_path),'wb') as f:
                f.setnchannels(wav.shape[0]); f.setsampwidth(2); f.setframerate(audio['sample_rate'])
                f.writeframes((wav.T.clip(-1,1)*32767).astype('<i2').tobytes())
            # Preserve the unquantized waveform separately, for audio case analysis.
            np.save(root/(name+'.audio.npy'),wav)
            av=root/(name+'.av.mp4')
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg,'-y','-i',str(video),'-i',str(wav_path),'-c:v','copy','-c:a','aac','-b:a','192k',str(av)],
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
            reader=iio.get_reader(str(video)); metadata=reader.get_meta_data(); decoded=sum(1 for _ in reader); reader.close()
            assert decoded==record['actual_frames']
            record.update(status='success',encode_write_s=time.perf_counter()-encode_start,
                          video_path=str(video),audio_path=str(wav_path),av_path=str(av),
                          actual_duration_s=record['actual_frames']/cfg['fps'],decoded_frame_count=decoded,
                          file_bytes=video.stat().st_size,video_metadata=metadata,
                          container_bitrate_bps=video.stat().st_size*8/(record['actual_frames']/cfg['fps']),
                          video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                          audio_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                          encoder=dict(imageio=imageio.__version__,imageio_ffmpeg=imageio_ffmpeg.__version__,
                                       ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),codec='libx264',quality=8,preset='medium',pixel_format='yuv420p',macro_block_size=1))
            atomic(target,record)
            print('ML_RESULT',name,record['generation_wall_s'],record['torch_peak_allocated_bytes'],flush=True)
        except Exception as e:
            record.update(status='failed',error=repr(e),stage_wall_s=stages,denoiser_call_shapes=calls)
            atomic(target.with_suffix('.failed.json'),record)
            raise
        finally:
            net.forward=original
        return {'ui':{'text':[str(target)]}}

NODE_CLASS_MAPPINGS={'MLMeasuredH3':MLMeasuredH3}
