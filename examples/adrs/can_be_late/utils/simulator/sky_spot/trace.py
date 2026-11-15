import json
import os


class Trace:

    def __init__(self, trace_data: dict):
        self.gap_seconds = trace_data['metadata']['gap_seconds']
        self.metadata = trace_data['metadata']
        self._data = trace_data['data']
        self._price = trace_data.get('prices', None)

    @classmethod
    def from_file(cls, trace_file: str):
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        trace = Trace(trace_data)
        return trace

    def __getitem__(self, index: int):
        return self._data[index]

    def get_price(self, index: int):
        if self._price is None:
            return None
        return self._price[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for item in self._data:
            yield item

    def toJSON(self):
        return json.dumps(self._trace_data)


class TraceDataset:

    def __init__(self, trace_folder: str):
        self.trace_folder = trace_folder

        for trace_file in os.listdir(self.trace_folder):
            trace = Trace.from_file(os.path.join(self.trace_folder,
                                                 trace_file))
            self.traces.append(trace)
        assert all(
            trace.gap_seconds == self.traces[0].gap_seconds
            for trace in self.traces), 'All traces must have the same time gap'
        self.gap_seconds = self.traces[0].gap_seconds

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, index: int):
        return self.traces[index]

    def __iter__(self):
        for trace in self.traces:
            yield trace
