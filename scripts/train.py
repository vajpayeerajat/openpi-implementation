import dataclasses
import functools
import logging
import platform
import pprint
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tensorboardX
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.hub_sync as _hub_sync
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_tensorboard(
    config: _config.TrainConfig, *, start_step: int, enabled: bool = True
) -> tensorboardX.SummaryWriter | None:
    if not enabled:
        return None

    log_dir = config.checkpoint_dir / "tensorboard"
    # On resume, drop events written after the restored step by the interrupted run, so the curves don't overlap.
    writer = tensorboardX.SummaryWriter(str(log_dir), purge_step=start_step or None)
    writer.add_text("config", f"```\n{pprint.pformat(dataclasses.asdict(config))}\n```", global_step=start_step)
    # Train and val loss on one chart ("Custom Scalars" tab). Unlike add_scalars, this reuses the main writer's
    # tags, so it adds no extra runs and respects purge_step on resume.
    writer.add_custom_scalars({"loss": {"train vs val": ["Multiline", ["train/loss", "val/loss"]]}})
    logging.info(f"TensorBoard logs: {log_dir}")
    return writer


def _camera_grid(observation: _model.Observation, num_samples: int = 5) -> np.ndarray:
    """Returns an [H, W, C] uint8 grid: one row per sample, cameras side by side."""
    images = [np.asarray(img) for img in observation.images.values()]
    rows = [np.concatenate([img[i] for img in images], axis=1) for i in range(min(num_samples, len(images[0])))]
    grid = np.concatenate(rows, axis=0)
    # Model images are float in [-1, 1].
    return ((grid + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None, lr_mult=config.model.get_lr_mult()
    )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    # Scalars the model sowed during the loss (e.g. DINO token RMS); popped so they don't end up in the params.
    intermediates = nnx.pop(model, nnx.Intermediate)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    for path, value in intermediates.flat_state().items():
        info["/".join(str(p) for p in path)] = jnp.mean(jnp.asarray(value.value))
    if getattr(config.model, "dino_fusion", "none") == "expert_stream":
        for name in ("dino_gate", "dino_attn_gate"):
            gate = jnp.tanh(getattr(model, name).value)
            info[f"{name}_mean"] = jnp.mean(gate)
            info[f"{name}_absmax"] = jnp.max(jnp.abs(gate))
        info["dino_grad_norm"] = optax.global_norm(grads.filter(nnx_utils.PathRegex("dino_.*")))
    return new_state, info


@at.typecheck
def val_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    # Evaluate the weights that get checkpointed: the EMA weights when EMA is enabled.
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    # train=False disables image augmentation.
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))


def run_validation(
    pval_step: Any,
    state: training_utils.TrainState,
    val_loader: _data_loader.DataLoader,
    mesh: jax.sharding.Mesh,
    seed: int,
) -> float:
    # A fixed rng per batch, so the flow-matching noise and timesteps are identical at every evaluation.
    rng = jax.random.key(seed)
    losses = []
    for i, batch in enumerate(val_loader):
        with sharding.set_mesh(mesh):
            losses.append(pval_step(jax.random.fold_in(rng, i), state, batch))
    return float(np.mean(jax.device_get(losses)))


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        # With Hub uploads every checkpoint lives on the Hub, so keep only the latest one locally (for --resume).
        keep_period=None if config.hub_repo_id else config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    hub = None
    if config.hub_repo_id:
        hub = _hub_sync.HubSync(
            config.hub_repo_id,
            tensorboard_dir=config.checkpoint_dir / "tensorboard" if config.tensorboard_enabled else None,
            private=config.hub_private,
            upload_train_state=config.hub_upload_train_state,
        )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    val_loader = None
    if config.val_repo_id is not None:
        # Few workers: the subset is small and the workers stay alive (and hold memory) between evaluations.
        val_loader = _data_loader.create_val_data_loader(
            config, sharding=data_sharding, num_workers=min(config.num_workers, 2)
        )
        logging.info(
            f"Validation on {config.val_repo_id}: {config.val_num_batches} batches every {config.val_interval} steps"
        )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pval_step = jax.jit(
        val_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    tb_writer = init_tensorboard(config, start_step=start_step, enabled=config.tensorboard_enabled)
    if tb_writer is not None:
        tb_writer.add_image("camera_views", _camera_grid(batch[0]), global_step=start_step, dataformats="HWC")
    lr_schedule = config.lr_schedule.create()

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # Wall-clock time and time spent waiting on the data loader since the last log, for throughput scalars.
    last_log_time = time.perf_counter()
    last_log_step = start_step
    data_time = 0.0
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            # Blocks on the device, so the wall-clock time below covers the finished steps.
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if tb_writer is not None:
                now = time.perf_counter()
                elapsed = now - last_log_time
                num_steps = max(step - last_log_step, 1)
                for key, value in reduced_info.items():
                    tb_writer.add_scalar(f"train/{key}", float(value), step)
                tb_writer.add_scalar("train/learning_rate", float(lr_schedule(step)), step)
                tb_writer.add_scalar("perf/steps_per_sec", num_steps / elapsed, step)
                tb_writer.add_scalar("perf/samples_per_sec", num_steps * config.batch_size / elapsed, step)
                tb_writer.add_scalar("perf/data_wait_fraction", data_time / elapsed, step)
                tb_writer.flush()
                last_log_time, last_log_step, data_time = now, step, 0.0
                if hub is not None and config.hub_tensorboard_interval and step % config.hub_tensorboard_interval == 0:
                    hub.upload_tensorboard(block=False)
            infos = []
        if val_loader is not None and (step % config.val_interval == 0 or step == config.num_train_steps - 1):
            val_start = time.perf_counter()
            val_loss = run_validation(pval_step, train_state, val_loader, mesh, config.seed)
            val_time = time.perf_counter() - val_start
            pbar.write(f"Step {step}: val_loss={val_loss:.4f} ({val_time:.1f}s)")
            wandb.log({"val_loss": val_loss}, step=step)
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, step)
                tb_writer.add_scalar("perf/val_seconds", val_time, step)
                tb_writer.flush()
            # Keep validation time out of the training throughput scalars.
            last_log_time += val_time
        data_start = time.perf_counter()
        batch = next(data_iter)
        data_time += time.perf_counter() - data_start

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            if hub is not None:
                # Saving deletes the previous local checkpoint, so its upload must have finished.
                hub.wait()
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if hub is not None:
                # Upload only fully written checkpoints.
                checkpoint_manager.wait_until_finished()
                hub.upload_checkpoint(config.checkpoint_dir / str(step), step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if tb_writer is not None:
        tb_writer.close()
    if hub is not None:
        logging.info("Waiting for Hub uploads to finish")
        hub.upload_tensorboard()
        hub.close()


if __name__ == "__main__":
    main(_config.cli())
