import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
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


def retain_review_checkpoints(run, steps):
    """Opt-in artifact retention after save; never stops or changes training.

    snapshot_steps.json is a list of positive cumulative decision thresholds.
    Preserve the first saved checkpoint at/above each target, without overwriting
    an existing snapshot. Archive failures are telemetry, not learner failures.
    """
    run = Path(run)
    plan = run/"snapshot_steps.json"
    if not plan.exists(): return {}
    result = {"milestone_archive_error":None}
    try:
        targets = json.loads(plan.read_text(encoding="utf-8"))
        if not isinstance(targets,list) or any(type(t) is not int or t < 1 for t in targets):
            raise ValueError("snapshot_steps.json must contain positive integer targets")
        for target in sorted(set(targets)):
            destination = run/f"milestone-{target}.pt"
            if steps < target or destination.exists(): continue
            temporary = None
            try:
                with (run/"latest.pt").open("rb") as source, tempfile.NamedTemporaryFile(
                        dir=run,prefix=f"milestone-{target}-",suffix=".tmp",delete=False) as copy:
                    temporary = Path(copy.name)
                    shutil.copyfileobj(source,copy)
                # Atomic no-clobber publication on the same filesystem.
                os.link(temporary,destination)
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
            metadata = dict(target_steps=target,actual_steps=steps,path=str(destination),sha256=sha256(destination))
            atomic_json(destination.with_suffix(".json"),metadata)
            result["milestone_last_saved"] = metadata
    except Exception as error:
        result["milestone_archive_error"] = f"{type(error).__name__}: {error}"
    return result
