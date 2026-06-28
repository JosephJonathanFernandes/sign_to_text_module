from collections import Counter, deque

class PredictionMomentum:
    """Simple momentum buffer: commit when a class appears >= commit_count

    Keeps recent (idx, conf) tuples in a circular buffer and commits a
    prediction when majority agreement, average confidence, and minimum
    occurrences are satisfied.
    """
    def __init__(self, window: int = 5, commit_count: int = 3, min_avg_conf: float = 0.6):
        self.window = window
        self.commit_count = commit_count
        self.min_avg_conf = min_avg_conf
        self._hist = deque(maxlen=window)

    def push(self, idx: int, conf: float) -> None:
        self._hist.append((int(idx), float(conf)))

    def get_commit(self):
        """Return (idx, avg_conf) if commit conditions met, else None."""
        if len(self._hist) < self.commit_count:
            return None
        counts = Counter([h[0] for h in self._hist])
        most, cnt = counts.most_common(1)[0]
        if cnt < self.commit_count:
            return None
        confs = [h[1] for h in self._hist if h[0] == most]
        avg_conf = sum(confs) / len(confs)
        if avg_conf < self.min_avg_conf:
            return None
        return int(most), float(avg_conf)

    def clear(self):
        self._hist.clear()
