import math
import statistics
import subprocess
import threading
from collections import UserDict, defaultdict
from pathlib import Path

import rich
import rich.table
from torch.utils.tensorboard import SummaryWriter


def launch_tensorboard(log_dir: Path, port: int | str = 6006) -> int:
    """Launch TensorBoard pointing to the given log directory, at the given port
    (http://localhost:port). Default is http://localhost:6006.

    Args:
        log_dir (Path): Directory where TensorBoard will look for logs.

    Returns:
        int: The return code from the TensorBoard subprocess (0 if successful).
    """
    cmd = [
        "tensorboard",
        "--logdir",
        str(log_dir),
        "--host",
        "localhost",
        "--port",
        str(port),
    ]
    return subprocess.call(cmd)  # noqa: S603


def load_tensorboard(log_dir: Path) -> None:
    """Load TensorBoard in a separate threads.

    Args:
        log_dir (Path): Directory where TensorBoard will look for logs.
    """
    t = threading.Thread(target=launch_tensorboard, args=(log_dir,))
    t.start()


class TensorboardLogger(UserDict):
    """
    A logger class that manages logging to TensorBoard and optionally to terminal.

    This class extends UserDict and provides functionality to buffer numeric values,
    calculate their means, and log them to TensorBoard's SummaryWriter. It supports
    both buffered logging (with automatic mean calculation) and direct logging.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance for logging metrics.

    Methods:
        log(step: int): Computes mean of buffered values, logs to TensorBoard,
            and clears buffer.
        direct_log(key: str, value: float, step: int): Logs a single value immediately
            without buffering.
        __setitem__(key: str, value: float): Adds a value to the buffer
            for the specified key.

    Example:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> writer = SummaryWriter()
        >>> logger = TensorboardLogger(writer)
        >>> logger['loss'] = 0.5
        >>> logger['loss'] = 0.3
        >>> logger.log(step=1)  # Logs mean loss (0.4) to TensorBoard
    """

    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer
        self._buffer = defaultdict(list[float])
        self.console = rich.console.Console()

    @staticmethod
    def _compute_confidence_interval(values: list[float], confidence=0.95):
        min_samples_for_stdev = 2
        mean = statistics.fmean(values)
        n = len(values)
        if len(values) < min_samples_for_stdev:
            return mean, 0.0

        stdev = statistics.stdev(values)
        z_score = math.erf(confidence) * math.sqrt(2)
        h = stdev * z_score / math.sqrt(n)
        return mean, h

    def _print_table(self, step: int, confidence=0.95) -> None:
        table = rich.table.Table(title=f"Step {step}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        for key, values in self._buffer.items():
            mean, ci = self._compute_confidence_interval(values, confidence)
            mean_pr = f"{mean:.4f}" if ci == 0.0 else f"{mean:.4f} ± {ci:.4f}"
            table.add_row(key, mean_pr)

        self.console.print(table)

    def log(self, step: int, *, verbose: bool = True) -> None:
        if not self._buffer:
            return

        parts = []
        for key, values in self._buffer.items():
            mean_value = statistics.fmean(values)
            self.writer.add_scalar(key, mean_value, step)
            parts.append(f"{key}: {mean_value:.4f}")
        if verbose:
            self._print_table(step)
        self._buffer.clear()

    def direct_log(
        self,
        key: str,
        value: float,
        step: int,
        *,
        verbose: bool = True,
    ) -> None:
        self.writer.add_scalar(key, value, step)
        if verbose:
            self.console.print(
                f"[bold blue][Step {step}][/bold blue] {key}: {value:.4f}",
            )

    def __setitem__(self, key: str, value: float) -> None:
        self._buffer[key].append(value)
