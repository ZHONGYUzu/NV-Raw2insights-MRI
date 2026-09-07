"""Stdlib-only compute-node resource record (psutil is absent in MRI env)."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args()
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True) if shutil.which('nvidia-smi') else None
mem=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None
disk=shutil.disk_usage(a.output.parent)
data=dict(timestamp=datetime.now(timezone.utc).isoformat(),host=platform.node(),
          cpu_count=os.cpu_count(),slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
          slurm_memory_mb=os.environ.get('SLURM_MEM_PER_NODE'),memory=mem,
          disk_free_bytes=disk.free,gpu=gpu.stdout if gpu else None,
          gpu_probe_stderr=gpu.stderr if gpu else None,python=platform.python_version())
a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)
