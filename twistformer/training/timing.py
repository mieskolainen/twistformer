# Diagnostic batch timing with completed device work
#
# mikael.mieskolainen@cern.ch (2026)

import csv
import time
from pathlib import Path
import torch


class BatchTiming:
    def __init__(self, count, device, path):
        """
        Buffer timing rows without writing to the run directory each batch

        Args:
            count:  Maximum measured batches, zero disables measurement
            device:  Training device
            path:  CSV destination for this rank
        """
        self.count, self.device, self.path = count, device, Path(path)
        self.rows, self.saved = [], 0
        self.metadata_ms = 0.0

    @property
    def active(self):
        """
        Check whether the measurement budget is still available

        Returns:
            active: More batches can be measured
        """
        return len(self.rows) < self.count

    def synchronize(self):
        """
        Wait for CUDA work only during diagnostic measurement
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def persist(self, timer):
        """
        Measure synchronous run metadata updates inside the training step

        Args:
            timer:  Training lifecycle timer
        """
        if not self.active:
            return timer.persist()
        started = time.perf_counter()
        try:
            return timer.persist()
        finally:
            self.metadata_ms += 1000 * (time.perf_counter() - started)

    def batches(self, loader, epoch):
        """
        Separate loader waits from steps that complete on the training device

        Args:
            loader:  Training batches on the host
            epoch:  Zero-based training epoch
        Yields:
            batch: Original batch without feature or weight changes
        """
        if not self.active:
            yield from loader
            return
        self.synchronize()
        started = time.perf_counter()
        iterator = iter(loader)
        for index in range(len(loader)):
            if not self.active:
                yield from iterator
                return
            batch = next(iterator)
            loaded = time.perf_counter()
            events, padded_tokens = batch.mask.shape
            valid_tokens = int(batch.mask.sum())
            self.metadata_ms = 0.0
            step_started = time.perf_counter()
            try:
                yield batch
            finally:
                self.synchronize()
                finished = time.perf_counter()
                self.rows.append({
                    "epoch": epoch + 1, "batch": index + 1,
                    "events": events, "padded_tokens": padded_tokens,
                    "valid_tokens": valid_tokens,
                    "padding_fraction": 1 - valid_tokens / (events * padded_tokens),
                    "data_wait_ms": 1000 * (loaded - started),
                    "step_ms": 1000 * (finished - step_started),
                    "metadata_ms": self.metadata_ms,
                })
                if not self.active:
                    self.save()
            # Exclude the diagnostic CSV write from the next loader wait
            started = time.perf_counter()

    def save(self):
        """
        Write buffered measurements at the budget or run boundary
        """
        if len(self.rows) == self.saved:
            return
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.saved = len(self.rows)
