"""Console and CSV logging."""

import csv
import os
import time
from typing import Dict, Optional


class RunLogger:
    def __init__(self, out_dir: str, name: str = "train", print_every: int = 20):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{name}.csv")
        self.print_every = print_every
        self.start = time.time()
        self._writer = None
        self._file = None

    def log(self, step: int, values: Dict[str, float], prefix: str = "") -> None:
        row = {"step": step, "elapsed_s": round(time.time() - self.start, 1)}
        row.update({f"{prefix}{k}": (round(v, 6) if isinstance(v, float) else v)
                    for k, v in values.items()})
        if self._writer is None:
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            if self._file.tell() == 0:
                self._writer.writeheader()
        # a run that adds a column mid-flight would otherwise raise
        for k in row:
            if k not in self._writer.fieldnames:
                row = {k2: v for k2, v in row.items() if k2 in self._writer.fieldnames}
                break
        self._writer.writerow(row)
        self._file.flush()

        if step % self.print_every == 0:
            body = "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in values.items())
            print(f"[{step:>7}] {body}", flush=True)

    def close(self) -> None:
        if self._file:
            self._file.close()
