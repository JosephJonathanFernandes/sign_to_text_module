"""
Real-time profiler for sign language inference pipeline.
Uses time.perf_counter() for accurate measurements (no estimation).
Tracks: frame timings, bottlenecks, FPS, inference frequency.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics


@dataclass
class TimingStats:
    """Statistics for a single profiled section."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent_times: deque = None  # deque of last 100 measurements
    
    def __post_init__(self):
        if self.recent_times is None:
            self.recent_times = deque(maxlen=100)
    
    @property
    def avg_ms(self) -> float:
        """Average timing in milliseconds."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count
    
    @property
    def median_ms(self) -> float:
        """Median timing in milliseconds."""
        if not self.recent_times:
            return 0.0
        return statistics.median(self.recent_times)
    
    def add_measurement(self, ms: float):
        """Record a new timing measurement."""
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        self.recent_times.append(ms)


class InferenceProfiler:
    """Real-time profiler for inference pipeline (uses time.perf_counter())."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize profiler.
        
        Args:
            max_history: Keep rolling statistics for last N frames
        """
        self.stats: Dict[str, TimingStats] = defaultdict(lambda: TimingStats(name=''))
        self.frame_times: deque = deque(maxlen=max_history)
        self.frame_count = 0
        self.inference_count = 0
        self.total_elapsed = 0.0
        self._frame_start_time = 0.0
        self._session_start_time = time.perf_counter()
    
    def start_frame(self):
        """Mark the start of a frame timing."""
        self._frame_start_time = time.perf_counter()
    
    def end_frame(self):
        """Mark the end of a frame, record total frame time."""
        elapsed_sec = time.perf_counter() - self._frame_start_time
        elapsed_ms = elapsed_sec * 1000.0
        self.frame_times.append(elapsed_ms)
        self.frame_count += 1
        self.total_elapsed += elapsed_sec
    
    def start_section(self, name: str):
        """Mark the start of a named section."""
        return self._SectionContext(self, name)
    
    def record_section(self, name: str, elapsed_ms: float):
        """Directly record a section timing."""
        if name not in self.stats:
            self.stats[name] = TimingStats(name=name)
        self.stats[name].add_measurement(elapsed_ms)
    
    def record_inference(self):
        """Record that an inference occurred."""
        self.inference_count += 1
    
    def get_fps(self) -> float:
        """Calculate current FPS based on recent frames."""
        if not self.frame_times or len(self.frame_times) < 2:
            return 0.0
        avg_frame_ms = statistics.mean(self.frame_times)
        if avg_frame_ms == 0:
            return 0.0
        return 1000.0 / avg_frame_ms
    
    def get_inference_frequency(self) -> float:
        """Calculate inferences per second."""
        if self.total_elapsed == 0:
            return 0.0
        return self.inference_count / self.total_elapsed
    
    def get_average_frame_time(self) -> float:
        """Get average frame time in milliseconds."""
        if not self.frame_times:
            return 0.0
        return statistics.mean(self.frame_times)
    
    def get_percentile_frame_time(self, percentile: float) -> float:
        """Get frame time at given percentile (0-100)."""
        if not self.frame_times or len(self.frame_times) < 2:
            return 0.0
        sorted_times = sorted(self.frame_times)
        idx = int(len(sorted_times) * percentile / 100.0)
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx]
    
    def get_bottleneck_ranking(self) -> List[tuple]:
        """
        Get sections ranked by average time.
        Returns: [(rank, name, avg_ms, % of frame), ...]
        """
        avg_frame_time = self.get_average_frame_time()
        if avg_frame_time == 0:
            return []
        
        # Calculate percentage of frame time for each section
        section_totals = []
        for name, stat in self.stats.items():
            pct = (stat.avg_ms / avg_frame_time) * 100.0
            section_totals.append((name, stat.avg_ms, pct, stat.count))
        
        # Sort by average time descending
        section_totals.sort(key=lambda x: x[1], reverse=True)
        
        return [(i + 1, name, avg_ms, pct) for i, (name, avg_ms, pct, _) in enumerate(section_totals)]
    
    def print_report(self, title: str = "Profiling Report"):
        """Print a detailed profiling report."""
        print("\n" + "=" * 90)
        print(f"  {title} [Frame {self.frame_count}]")
        print("=" * 90)
        
        # FPS and frame statistics
        avg_frame_ms = self.get_average_frame_time()
        fps = self.get_fps()
        p95_frame_ms = self.get_percentile_frame_time(95)
        p99_frame_ms = self.get_percentile_frame_time(99)
        infer_freq = self.get_inference_frequency()
        
        print(f"\n{'FRAME STATISTICS':^90}")
        print("-" * 90)
        print(f"  Total Frames:              {self.frame_count}")
        print(f"  Average Frame Time:        {avg_frame_ms:8.2f} ms")
        print(f"  Median Frame Time:         {statistics.median(self.frame_times) if self.frame_times else 0:8.2f} ms")
        print(f"  P95 Frame Time:            {p95_frame_ms:8.2f} ms")
        print(f"  P99 Frame Time:            {p99_frame_ms:8.2f} ms")
        print(f"  Min Frame Time:            {min(self.frame_times) if self.frame_times else 0:8.2f} ms")
        print(f"  Max Frame Time:            {max(self.frame_times) if self.frame_times else 0:8.2f} ms")
        print(f"  Target Frame Budget (30fps): 33.33 ms")
        
        budget_util = (avg_frame_ms / 33.33) * 100.0
        print(f"  Budget Utilization:        {budget_util:8.1f}%")
        
        print(f"\n  Current FPS:               {fps:8.2f}")
        print(f"  Total Inferences:          {self.inference_count}")
        print(f"  Inference Frequency:       {infer_freq:8.2f} inferences/sec")
        if self.inference_count > 0:
            avg_frames_per_inf = self.frame_count / self.inference_count
            print(f"  Avg Frames Per Inference:  {avg_frames_per_inf:8.2f}")
        
        # Bottleneck ranking
        ranking = self.get_bottleneck_ranking()
        if ranking:
            print(f"\n{'BOTTLENECK RANKING (by average time)':^90}")
            print("-" * 90)
            print(f"{'Rank':<6} {'Component':<35} {'Avg (ms)':<12} {'% Frame':<12} {'Count':<10}")
            print("-" * 90)
            
            total_accounted = 0.0
            for rank, name, avg_ms, pct in ranking:
                print(f"  {rank:<4} {name:<33} {avg_ms:>10.2f}  {pct:>10.1f}%  {self.stats[name].count:>8}")
                total_accounted += pct
            
            print("-" * 90)
            print(f"{'  TOTAL ACCOUNTED':^55} {total_accounted:>10.1f}%")
            unaccounted = 100.0 - total_accounted
            print(f"{'  Unaccounted (overhead/profiler):':^55} {unaccounted:>10.1f}%")
        
        print("=" * 90 + "\n")
    
    def print_section_details(self):
        """Print detailed statistics for each section."""
        print(f"\n{'SECTION DETAILS':^90}")
        print("-" * 90)
        print(f"{'Section':<35} {'Count':<8} {'Avg(ms)':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 90)
        
        for name, stat in sorted(self.stats.items(), key=lambda x: x[1].avg_ms, reverse=True):
            print(
                f"  {name:<33} {stat.count:<8} {stat.avg_ms:>8.2f}  "
                f"{stat.min_ms:>8.2f}  {stat.max_ms:>8.2f}  {stat.median_ms:>8.2f}"
            )
        
        print("-" * 90 + "\n")
    
    class _SectionContext:
        """Context manager for timing a section."""
        
        def __init__(self, profiler: 'InferenceProfiler', name: str):
            self.profiler = profiler
            self.name = name
            self.start_time = 0.0
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_sec = time.perf_counter() - self.start_time
            elapsed_ms = elapsed_sec * 1000.0
            
            if self.name not in self.profiler.stats:
                self.profiler.stats[self.name] = TimingStats(name=self.name)
            
            self.profiler.stats[self.name].add_measurement(elapsed_ms)
            return False


# Global profiler instance
_global_profiler: Optional[InferenceProfiler] = None


def get_profiler() -> InferenceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = InferenceProfiler()
    return _global_profiler


def profile_section(name: str):
    """Context manager for profiling a named section."""
    return get_profiler().start_section(name)


def reset_profiler():
    """Reset the global profiler."""
    global _global_profiler
    _global_profiler = InferenceProfiler()


# Convenience functions
def start_frame():
    """Mark the start of a frame."""
    get_profiler().start_frame()


def end_frame():
    """Mark the end of a frame."""
    get_profiler().end_frame()


def record_inference():
    """Record that an inference occurred."""
    get_profiler().record_inference()


def print_report(title: str = "Profiling Report"):
    """Print the profiling report."""
    get_profiler().print_report(title=title)


def print_details():
    """Print detailed section statistics."""
    get_profiler().print_section_details()
