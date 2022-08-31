import torch
from timeit import default_timer

class Timer:
    def __init__(self, device=None):
        self.timer = default_timer
        if device is None:
            self.device = get_bench_device()
        else:
            self.device = device

    def __enter__(self):
        if self.device == 'cuda:0':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == 'cuda:0':
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic
