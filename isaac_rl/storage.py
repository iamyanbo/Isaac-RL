import hashlib
import io
import json
import os
from pathlib import Path
import time

import torch


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01)


def checkpoint(path, model, optimizer, metadata):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),**metadata),temp)
    os.replace(temp,path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_snapshot(path, device="cpu"):
    """Hash and deserialize the SAME bytes, even if a trainer replaces latest.pt."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload),map_location=device,weights_only=False)
    return saved,digest,payload


def source_fingerprint(root):
    return {str(path.relative_to(root)):sha256(path) for directory in ("isaac_rl","mod")
            for path in (root/directory).rglob("*") if path.suffix in (".py",".lua",".xml")}


def capture_failure_states(path, envs):
    """Bounded current native evidence; never replace the optimized checkpoint.

    env.step assigns its raw response before observation validation, so this
    retains a bad frame that never made it into the encoder/history/checkpoint.
    A capture failure must not mask the original training exception.
    """
    try:
        atomic_json(path,[env.state for env in envs])
        return {"failure_states":str(path)}
    except Exception as error:
        return {"failure_capture_error":f"{type(error).__name__}: {error}"}
