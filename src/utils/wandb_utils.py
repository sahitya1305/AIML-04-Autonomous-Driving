"""Weights & Biases wrapper that never crashes the project.

* Disabled by default. Enable with `wandb.enabled: true` in the config or the `--wandb` flag.
* The API key is read by the wandb library from the WANDB_API_KEY environment variable (or a previous `wandb login`).
  Nothing is ever stored in the repository.
* If wandb is not installed, or no credentials exist and mode is `online`, we fall back to OFFLINE mode (runs are saved
  locally in ./wandb and can be uploaded later with `wandb sync`) and say so.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log = get_logger("wandb")


def has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    for f in (Path.home() / ".netrc", Path.home() / "_netrc"):
        try:
            if f.exists() and "api.wandb.ai" in f.read_text(errors="ignore"):
                return True
        except OSError:
            pass
    return False


def resolve_mode(requested: str, creds: bool) -> str:
    """online without credentials -> offline (so a missing key can never block or crash a run)."""
    if requested == "online" and not creds:
        return "offline"
    return requested


class WandbLogger:
    def __init__(self, enabled: bool = False, project: str = "aiml04-driving-perception", entity: Optional[str] = None,
                 run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, mode: str = "online",
                 job_type: str = "run", tags: Optional[list] = None):
        self.run = None
        self.wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            log.warning("wandb requested but not installed (pip install wandb). Logging disabled.")
            return
        eff = resolve_mode(mode, has_credentials())
        if eff != mode:
            log.warning("No WANDB_API_KEY / login found: using OFFLINE mode. Set the key and use `wandb sync` to upload.")
        try:
            self.run = wandb.init(project=project, entity=entity, name=run_name, config=config or {}, mode=eff,
                                  job_type=job_type, tags=tags or ["aiml-04"], reinit=True)
            self.wandb = wandb
            log.info("W&B run started (%s mode).", eff)
        except Exception as e:  # network/auth problems must not kill a long job
            log.warning("W&B init failed (%s). Logging disabled.", e)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], job_type: str = "run", run_name: Optional[str] = None,
                    extra_config: Optional[Dict[str, Any]] = None) -> "WandbLogger":
        w = cfg.get("wandb", {})
        conf = {k: v for k, v in cfg.items() if k != "wandb"}
        conf.update(extra_config or {})
        return cls(w.get("enabled", False), w.get("project", "aiml04-driving-perception"), w.get("entity"),
                   run_name, conf, w.get("mode", "online"), job_type)

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if self.active:
            self.wandb.log(data, step=step)

    def summary(self, data: Dict[str, Any]):
        if self.active:
            for k, v in data.items():
                self.run.summary[k] = v

    def log_image(self, key: str, image_bgr, caption: str = ""):
        if self.active:
            import cv2
            self.wandb.log({key: self.wandb.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=caption)})

    def log_video(self, key: str, path, fps: int = 20):
        if self.active:
            try:
                self.wandb.log({key: self.wandb.Video(str(path), fps=fps, format="mp4")})
            except Exception as e:
                log.warning("Could not upload video to W&B: %s", e)

    def log_file(self, path):
        if self.active and Path(path).exists():
            self.wandb.save(str(path), policy="now")

    def finish(self):
        if self.active:
            self.run.finish()
            self.run = None
