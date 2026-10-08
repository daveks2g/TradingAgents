"""Data pipeline: reads bar CSV incrementally and maintains a rolling buffer."""

import time
from collections import deque
from pathlib import Path

import pandas as pd

import config


class BarBuffer:
    """Rolling buffer of recent bars with incremental CSV reading."""

    def __init__(self, csv_path: Path = config.CSV_PATH, max_bars: int = config.LOOKBACK_WINDOW):
        self.csv_path = csv_path
        self.max_bars = max_bars
        self.bars: deque[dict] = deque(maxlen=max_bars)
        self._byte_offset: int = 0   # byte position after last read
        self._header: list[str] | None = None
        self._preload_existing()

    def _preload_existing(self):
        """Load existing bars into buffer silently, advancing the read cursor.

        This prevents re-processing old bars on startup while still having
        historical data available for feature computation.
        """
        if not self.csv_path.exists():
            return
        try:
            rows = self._read_new_rows()
            for row in rows:
                self.bars.append(row)
        except Exception:
            pass  # File may not exist yet or be empty

    def _read_new_rows(self) -> list[dict]:
        """Read only new rows appended since last read, using byte-offset seek."""
        rows = []
        try:
            with open(self.csv_path, "r", encoding="utf-8", errors="replace") as f:
                # Handle header
                if self._header is None:
                    header_line = f.readline()
                    if not header_line:
                        return []
                    self._header = [c.strip() for c in header_line.strip().split(",")]
                    self._byte_offset = f.tell()
                else:
                    # Detect file truncation/recreation (byte_offset past end of file)
                    f.seek(0, 2)  # seek to end
                    file_size = f.tell()
                    if self._byte_offset > file_size:
                        # File was recreated — reset and re-read header
                        f.seek(0)
                        header_line = f.readline()
                        if not header_line:
                            return []
                        self._header = [c.strip() for c in header_line.strip().split(",")]
                        self._byte_offset = f.tell()
                    else:
                        f.seek(self._byte_offset)

                # Read only new lines from current position
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    values = stripped.split(",")
                    if len(values) != len(self._header):
                        continue
                    row = {}
                    for col, val in zip(self._header, values):
                        try:
                            row[col] = float(val)
                        except ValueError:
                            row[col] = val
                    rows.append(row)

                self._byte_offset = f.tell()
        except FileNotFoundError:
            pass
        return rows

    def poll(self) -> list[dict]:
        """Check for new bars. Returns list of new rows (NOT yet added to buffer)."""
        return self._read_new_rows()

    def add(self, bar: dict) -> None:
        """Add a single bar to the rolling buffer."""
        self.bars.append(bar)

    def to_dataframe(self) -> pd.DataFrame:
        """Return current buffer as a DataFrame."""
        return pd.DataFrame(list(self.bars))

    def __len__(self) -> int:
        return len(self.bars)


def stream_bars(csv_path: Path = config.CSV_PATH, poll_interval: float = config.POLL_INTERVAL_SEC):
    """Generator that yields new bars as they appear in the CSV."""
    buf = BarBuffer(csv_path)
    while True:
        new_bars = buf.poll()
        for bar in new_bars:
            buf.add(bar)
            yield bar, buf
        if not new_bars:
            time.sleep(poll_interval)
