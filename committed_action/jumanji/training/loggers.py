from __future__ import annotations

import abc
import collections
import inspect
import logging
import os
import pickle
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, DefaultDict, Dict, Optional, Type

import jax.numpy as jnp
import numpy as np
import omegaconf
import tensorboardX
import wandb


class Logger(AbstractContextManager):
    def __init__(
        self,
        save_checkpoint: bool,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ):
        self.save_checkpoint = save_checkpoint
        self.checkpoint_file_name = checkpoint_file_name
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        """Write a dictionary of metrics to the logger.

        Args:
            data: dictionary of metrics names and their values.
            label: optional label (e.g. 'train' or 'eval').
            env_steps: optional env step count.
        """

    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    def upload_checkpoint(self, checkpoint_path: str) -> None:
        """Uploads a checkpoint when exiting the logger or during training."""

    def is_loggable(self, value: Any) -> bool:
        """Returns True if the value is loggable."""
        if isinstance(value, (float, int)):
            return True
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            return bool(value.ndim == 0)
        return False

    def __enter__(self) -> Logger:
        logging.info("Starting logger.")
        self._variables_enter = self._get_variables()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.save_checkpoint:
            self._variables_exit = self._get_variables()
            self._save_and_upload_checkpoint()
        logging.info("Closing logger...")
        self.close()

    def save_checkpoint_now(
        self,
        training_state: Any,
        checkpoint_name: Optional[str] = None,
    ) -> None:
        """Save and upload a checkpoint immediately."""
        if not self.save_checkpoint:
            return

        checkpoint_name = checkpoint_name or self.checkpoint_file_name
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        logging.info(f"Saving checkpoint to '{checkpoint_path}'...")
        with open(checkpoint_path, "wb") as file_:
            pickle.dump(training_state, file_)

        self.upload_checkpoint(checkpoint_path)
        logging.info(f"Checkpoint saved at '{checkpoint_path}'.")

    def _save_and_upload_checkpoint(self) -> None:
        """Grabs the `training_state` variable from within the context manager,
        pickles it and saves it.
        """
        logging.info("Saving final checkpoint...")
        in_context_variables = dict(set(self._variables_exit).difference(self._variables_enter))
        variable_id = in_context_variables.get("training_state", None)
        if variable_id is not None:
            training_state = self._variables_exit[("training_state", variable_id)]
        else:
            training_state = None
            logging.debug(
                "Logger did not find variable 'training_state' at the context manager level."
            )

        self.save_checkpoint_now(
            training_state=training_state,
            checkpoint_name=f"{self.checkpoint_file_name}_final.pkl",
        )

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the
        context manager.
        """
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}


class NoOpLogger(Logger):
    """Does nothing. This logger is useful in the case of multi-node training where only the
    master node should log.
    """

    def __init__(self) -> None:
        super().__init__(save_checkpoint=False)

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        pass


class TerminalLogger(Logger):
    """Logs to terminal."""

    def __init__(
        self,
        name: Optional[str] = None,
        save_checkpoint: bool = False,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        super().__init__(
            save_checkpoint=save_checkpoint,
            checkpoint_file_name=checkpoint_file_name,
            checkpoint_dir=checkpoint_dir,
        )
        if name:
            logging.info(f"Experiment: {name}.")

    def _format_values(self, data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()}: "
            f"{(f'{value:,}' if isinstance(value, int) else f'{value:.3f}')}"
            for key, value in sorted(data.items())
            if self.is_loggable(value)
        )

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps is not None:
            env_steps_str = f"Env Steps: {env_steps:.2e} | "
        else:
            env_steps_str = ""
        label_str = f"{label.replace('_', ' ').title()} >> " if label else ""
        logging.info(label_str + env_steps_str + self._format_values(data))


class ListLogger(Logger):
    """Logs to a dictionary of histories as lists."""

    def __init__(
        self,
        save_checkpoint: bool = False,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        super().__init__(
            save_checkpoint=save_checkpoint,
            checkpoint_file_name=checkpoint_file_name,
            checkpoint_dir=checkpoint_dir,
        )
        self.history: DefaultDict = collections.defaultdict(list)

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        for key, value in data.items():
            if self.is_loggable(value):
                self.history[key].append(value)


class TensorboardLogger(Logger):
    """Logs to tensorboard. To view logs, run a command like:
    tensorboard --logdir jumanji/training/outputs/{date}/{time}/{name}/
    """

    def __init__(
        self,
        name: str,
        save_checkpoint: bool = False,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        super().__init__(
            save_checkpoint=save_checkpoint,
            checkpoint_file_name=checkpoint_file_name,
            checkpoint_dir=checkpoint_dir,
        )
        if name:
            logging.info(name)
        self.writer = tensorboardX.SummaryWriter(logdir=name)
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if self.is_loggable(metric) and not np.isnan(metric):
                self.writer.add_scalar(
                    tag=f"{prefix}/{key}",
                    scalar_value=metric,
                    global_step=int(self._env_steps),
                )

    def close(self) -> None:
        self.writer.close()


class NeptuneLogger(Logger):
    """Logs to the neptune.ai platform."""

    def __init__(
        self,
        name: str,
        project: str,
        cfg: omegaconf.DictConfig,
        save_checkpoint: bool = False,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ):
        super().__init__(
            save_checkpoint=save_checkpoint,
            checkpoint_file_name=checkpoint_file_name,
            checkpoint_dir=checkpoint_dir,
        )
        from neptune import new as neptune
        self.run = neptune.init_run(project=project, name=name)
        self.run["config"] = cfg
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[float] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if self.is_loggable(metric) and not np.isnan(metric):
                self.run[f"{prefix}/{key}"].log(
                    float(metric),
                    step=int(self._env_steps),
                    wait=True,
                )

    def close(self) -> None:
        self.run.stop()

    def upload_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_name = os.path.basename(checkpoint_path)
        self.run[f"checkpoint/{checkpoint_name}"].upload(checkpoint_path)

class WandbLogger(Logger):
    """Logs to Weights & Biases."""

    def __init__(
        self,
        name: str,
        project: str,
        cfg: omegaconf.DictConfig,
        entity: Optional[str] = None,
        save_checkpoint: bool = False,
        checkpoint_file_name: str = "training_state",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        super().__init__(
            save_checkpoint=save_checkpoint,
            checkpoint_file_name=checkpoint_file_name,
            checkpoint_dir=checkpoint_dir,
        )

        # Convert OmegaConf config to a plain container for W&B.
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True)

        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            config=config,
        )
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[float] = None,
    ) -> None:
        if env_steps is not None:
            self._env_steps = env_steps

        prefix = f"{label}/" if label else ""
        metrics = {}

        for key, metric in data.items():
            if self.is_loggable(metric) and not np.isnan(metric):
                metrics[f"{prefix}{key}"] = float(metric)

        if metrics:
            self.run.log(metrics, step=int(self._env_steps))

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()

    def upload_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_name = os.path.basename(checkpoint_path)

        artifact = wandb.Artifact(
            name=f"{self.run.id}-checkpoints",
            type="checkpoint",
        )
        artifact.add_file(checkpoint_path, name=checkpoint_name)
        self.run.log_artifact(artifact)