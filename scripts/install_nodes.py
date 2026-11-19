"""Install only our three measurement nodes; never overwrite existing nodes."""
import argparse
from pathlib import Path
import shutil

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--comfy-dir',type=Path,required=True)
a=ap.parse_args()
assert (a.comfy_dir/'main.py').is_file(), 'Expected a ComfyUI checkout.'
for src in (Path(__file__).resolve().parents[1]/'custom_nodes').iterdir():
    target=a.comfy_dir/'custom_nodes'/src.name
    assert not target.exists(),f'Already installed: {target}'
    shutil.copytree(src,target,ignore=shutil.ignore_patterns('__pycache__'))
    print('Installed',target)
