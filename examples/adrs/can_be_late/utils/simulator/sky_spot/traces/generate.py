import argparse
import json
import os
import pathlib
import random
from typing import Dict, Type

import numpy as np

GENERATORS: Dict[str, Type['TraceGenerator']] = {}


class TraceGenerator:
    NAME = 'abstract'

    def __init__(self, trace_folder: str):
        self.trace_folder = pathlib.Path(trace_folder)
        self.trace_folder.mkdir(parents=True, exist_ok=True)

    def metadata(self):
        raise NotImplementedError

    def __init_subclass__(cls) -> None:
        GENERATORS[cls.NAME] = cls

    def generate(self, num_traces: int):
        raise NotImplementedError


class PoissonTraceGenerator(TraceGenerator):
    NAME = 'poisson'

    def __init__(self, trace_folder: str, gap_seconds: int, length: int, *,
                 hourly_rate: float, **kwargs):
        super().__init__(trace_folder)
        self.gap_seconds = gap_seconds
        self.hourly_rate = hourly_rate

        # Calculate the rate per gap seconds based on poisson distribution
        # https://math.stackexchange.com/questions/2480542/probability-of-an-event-occurring-within-a-smaller-time-interval-if-one-knows-th
        occurence_per_hour = -np.log(1 - self.hourly_rate)  # lambda
        self.gap_rate = 1 - np.exp(
            -occurence_per_hour * gap_seconds /
            3600)  # 1 - e^(-lambda * gap_seconds / 3600)
        print(
            f'hourly_rate: {self.hourly_rate:.2f}, gap_rate ({gap_seconds/3600:.2f} hours): {self.gap_rate:.2f}'
        )

        self.length = length
        self.output_folder = self.trace_folder / f'gap_{self.gap_seconds}-hourly_rate_{self.hourly_rate}'
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder}")

    def generate(self, num_traces: int):
        for i in range(num_traces):
            random.seed(i)
            file_path = self.output_folder / f'{i}.json'
            if file_path.exists():
                continue
            data = [
                int(random.random() < self.gap_rate)
                for _ in range(self.length)
            ]
            trace = {
                'generator': self.NAME,
                'metadata': self.metadata(),
                'data': data
            }
            with file_path.open('w') as f:
                json.dump(trace, f)

    def metadata(self):
        return {
            'gap_seconds': self.gap_seconds,
            'hourly_rate': self.hourly_rate,
            'length': self.length,
        }


class TwoExponentialGenerator(TraceGenerator):
    NAME = 'two_exp'

    def __init__(self,
                 trace_folder: str,
                 gap_seconds: int,
                 length: int,
                 *,
                 alive_scale: float,
                 wait_scale: float,
                 output_folder=None,
                 **kwargs):
        super().__init__(trace_folder)
        self.gap_seconds = gap_seconds
        self.length = length

        alive_lambda = 1 / alive_scale
        wait_lambda = 1 / wait_scale
        self.alive_lambda = alive_lambda
        self.wait_lambda = wait_lambda

        if output_folder is None:
            self.output_folder = self.trace_folder / f'gap_{self.gap_seconds}-alive_{alive_lambda}-wait_{wait_lambda}'
        else:
            self.output_folder = self.trace_folder / output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder}")

    def generate(self, num_traces: int):
        for i in range(num_traces):
            np.random.seed(i)
            random.seed(i)

            file_path = self.output_folder / f'{i}.json'
            alive_exp_dist = np.random.exponential(scale=1 / self.alive_lambda,
                                                   size=self.length)
            wait_exp_dist = np.random.exponential(scale=1 / self.wait_lambda,
                                                  size=self.length)
            data = []
            cnt = 0
            alive_first = random.random() < 0.5
            remaining = self.length
            if alive_first:
                while remaining > 0:
                    for i in range(int(np.ceil(alive_exp_dist[cnt]))):
                        data.append(0)
                        remaining -= 1
                    for j in range(int(np.ceil(wait_exp_dist[cnt]))):
                        data.append(1)
                        remaining -= 1
                    cnt += 1
            else:
                while remaining > 0:
                    for j in range(int(np.ceil(wait_exp_dist[cnt]))):
                        data.append(1)
                        remaining -= 1
                    for i in range(int(np.ceil(alive_exp_dist[cnt]))):
                        data.append(0)
                        remaining -= 1
                    cnt += 1
            trace = {
                'generator': self.NAME,
                'metadata': self.metadata(),
                'data': data
            }
            with file_path.open('w') as f:
                json.dump(trace, f)

    def metadata(self):
        return {
            'gap_seconds': self.gap_seconds,
            'alive_lambda': self.alive_lambda,
            'wait_lambda': self.wait_lambda,
            'length': self.length,
        }


class TwoGammaGenerator(TraceGenerator):
    NAME = 'two_gamma'

    def __init__(self, trace_folder: str, gap_seconds: int, length: int, *,
                 alive_a: float, alive_scale: float, wait_a: float,
                 wait_scale: float, seed: int, **kwargs):
        super().__init__(trace_folder)
        self.gap_seconds = gap_seconds
        self.length = length

        self.alive_alpha = alive_a
        self.alive_beta = 1 / alive_scale
        self.wait_alpha = wait_a
        self.wait_beta = 1 / wait_scale
        self.seed = seed

        self.output_folder = self.trace_folder / f'gap_{self.gap_seconds}-alive_{self.alive_alpha}_{self.alive_beta}-wait_{self.wait_alpha}_{self.wait_beta}'
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder}")

    def generate(self, num_traces: int):
        from scipy.stats import gamma
        for i in range(num_traces):
            np.random.seed(seed=i * self.seed + self.seed)
            random.seed(i * self.seed + self.seed)

            file_path = self.output_folder / f'{i}.json'
            alive_gamma_dist = gamma.rvs(a=self.alive_alpha,
                                         scale=1 / self.alive_beta,
                                         size=self.length)
            wait_gamma_dist = gamma.rvs(a=self.wait_alpha,
                                        scale=1 / self.wait_beta,
                                        size=self.length)
            data = []
            cnt = 0
            alive_first = random.random() < 0.5
            remaining = self.length
            if alive_first:
                while remaining > 0:
                    for i in range(int(np.ceil(alive_gamma_dist[cnt]))):
                        data.append(0)
                        remaining -= 1
                    for j in range(int(np.ceil(wait_gamma_dist[cnt]))):
                        data.append(1)
                        remaining -= 1
                    cnt += 1
            else:
                while remaining > 0:
                    for j in range(int(np.ceil(wait_gamma_dist[cnt]))):
                        data.append(1)
                        remaining -= 1
                    for i in range(int(np.ceil(alive_gamma_dist[cnt]))):
                        data.append(0)
                        remaining -= 1
                    cnt += 1
            trace = {
                'generator': self.NAME,
                'metadata': self.metadata(),
                'data': data
            }
            with file_path.open('w') as f:
                json.dump(trace, f)

    def metadata(self):
        return {
            'gap_seconds': self.gap_seconds,
            'alive_alpha': self.alive_alpha,
            'alive_beta': self.alive_beta,
            'wait_alpha': self.wait_alpha,
            'wait_beta': self.wait_beta,
            'length': self.length,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-folder', type=str, default='data/poisson')
    parser.add_argument('--generator', type=str, default='poisson')
    parser.add_argument('--gap-seconds', type=int, default=20 * 60)
    parser.add_argument(
        '--hourly-rate',
        type=float,
        default=0.1,
        help='Hourly probability of occurrence for Poisson generator')
    parser.add_argument('--length', type=int, default=60 * 24 * 7)
    parser.add_argument('--num-traces', type=int, default=100)
    parser.add_argument('--alive-a', type=float, default=2.3491)
    parser.add_argument('--alive-scale', type=float, default=2.3491)
    parser.add_argument('--wait-a', type=float, default=2.2745)
    parser.add_argument('--wait-scale', type=float, default=2.2745)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    generator_args = vars(args)
    generator_name = generator_args.pop('generator')
    num_traces = generator_args.pop('num_traces')
    trace_folder = generator_args.pop('trace_folder')
    gap_seconds = generator_args.pop('gap_seconds')
    length = generator_args.pop('length')

    GeneratorCls = GENERATORS[generator_name]

    generator = GeneratorCls(trace_folder, gap_seconds, length,
                             **generator_args)

    generator.generate(num_traces)
