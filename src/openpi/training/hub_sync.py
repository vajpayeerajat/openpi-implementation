"""Uploads training checkpoints and TensorBoard logs to a Hugging Face Hub repo in the background.

Layout in the repo:
    checkpoints/<step>/params/...      inference weights (what serving and eval scripts load)
    checkpoints/<step>/assets/...      norm stats
    checkpoints/<step>/train_state/... optimizer state, only with `upload_train_state` (needed to resume elsewhere)
    runs/...                           TensorBoard event files
"""

from concurrent import futures
import logging
import pathlib
import time

import huggingface_hub

# Files the checkpoint manager writes next to the items; small, and needed to load the step dir.
_ALWAYS_UPLOAD = ["_CHECKPOINT_METADATA", "params/**", "assets/**"]


class HubSync:
    def __init__(
        self,
        repo_id: str,
        *,
        tensorboard_dir: pathlib.Path | None,
        private: bool = True,
        upload_train_state: bool = False,
        retries: int = 3,
    ):
        self._repo_id = repo_id
        self._tensorboard_dir = tensorboard_dir
        self._upload_train_state = upload_train_state
        self._retries = retries
        self._api = huggingface_hub.HfApi()
        # Fails fast (before training starts) if not logged in or the repo name is not allowed.
        url = self._api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
        logging.info(f"Uploading checkpoints and TensorBoard logs to {url}")
        # One upload at a time, in order.
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="hub_sync")
        self._pending: futures.Future | None = None

    def upload_checkpoint(self, step_dir: pathlib.Path, step: int) -> None:
        """Starts uploading a finished checkpoint step dir (and the TensorBoard logs) in the background."""
        self.wait()
        self._pending = self._executor.submit(self._upload_checkpoint, pathlib.Path(step_dir), step)

    def upload_tensorboard(self, *, block: bool = True) -> None:
        """Uploads the TensorBoard logs. With block=False this never waits: if an upload is still running, it is
        skipped (a running checkpoint upload uploads the logs when it finishes), and failures are not retried."""
        if not block:
            if self._pending is not None and not self._pending.done():
                return
            self._pending = self._executor.submit(self._upload_tensorboard, retries=1)
            return
        self.wait()
        self._pending = self._executor.submit(self._upload_tensorboard)

    def wait(self) -> None:
        """Blocks until the previous upload has finished. Upload errors are logged, not raised."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        self.wait()
        self._executor.shutdown()

    def _upload_checkpoint(self, step_dir: pathlib.Path, step: int) -> None:
        allow = [*_ALWAYS_UPLOAD, *(["train_state/**"] if self._upload_train_state else [])]
        self._with_retries(
            f"checkpoint {step}",
            lambda: self._api.upload_folder(
                repo_id=self._repo_id,
                folder_path=step_dir,
                path_in_repo=f"checkpoints/{step}",
                allow_patterns=allow,
                commit_message=f"Checkpoint at step {step}",
            ),
        )
        self._upload_tensorboard()

    def _upload_tensorboard(self, retries: int | None = None) -> None:
        if self._tensorboard_dir is None or not self._tensorboard_dir.exists():
            return
        self._with_retries(
            "TensorBoard logs",
            lambda: self._api.upload_folder(
                repo_id=self._repo_id,
                folder_path=self._tensorboard_dir,
                path_in_repo="runs",
                commit_message="Update TensorBoard logs",
            ),
            retries=retries,
        )

    def _with_retries(self, what: str, upload, retries: int | None = None) -> None:
        retries = retries or self._retries
        for attempt in range(1, retries + 1):
            start = time.monotonic()
            try:
                upload()
                logging.info(f"Uploaded {what} to {self._repo_id} in {time.monotonic() - start:.0f}s")
                return
            except Exception as e:  # A failed upload must not stop training.
                logging.warning(f"Upload of {what} failed (attempt {attempt}/{retries}): {e!r}")
                if attempt < retries:
                    time.sleep(30 * attempt)
        logging.error(f"Giving up on uploading {what} to {self._repo_id}; it is not on the Hub.")
