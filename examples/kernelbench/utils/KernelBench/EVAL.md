# Evaluation
[WIP] More notes on Benchmarking Guide
To be updated more comprehensively with the benchmarking guide (ongoing PRs) & blog that we have been working on this quarter.

You should be **extra CAREFUL!** , be always paranoid about suspiciously good results — kernel engineers and existing compilers are already pretty good, so a >2x speedup for anything is highly unlikely. 


> “if you beat cudnn by more than 10%, think again” -- 
from [itsclivetime](https://x.com/itsclivetime/status/1992155951630307633?s=46)


If the model can reward hack, it will find ways to reward hack! This can especially happen during RL training or evolutionary search.

Check out resources here:
- KernelBench [v0.1 Release](https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/) 
- Cognition and Stanford's [Kevin](https://arxiv.org/abs/2507.11948) project on various hacking behaviors observed in RL training
- Jiwei Li's awesome [blogpost 1](https://deep-reinforce.com/defense_kernel_hack.html) and [blogpost 2](https://deep-reinforce.com/correctness_check.html) on Hacks and Defenses in Automatic GPU Kernel Generations

Our ongoing blogpost and PRs try to systematize and list out these behaviors and provide tests, detection, and mitigation toolings.

**Disclaimer**: KernelBench is an open-source evaluation framework. Due to limited bandwidth, the KernelBench team does not inspect, validate, or endorse any third-party kernels or reported results. Users are welcome to use the software infrastructure for evaluation, but should independently verify all results.


## Methodology
More on that coming.

To ensure **consistency and reproducibility**, we recommend using `modal` and we have provided / are adding more various modal cloud functions to standardize the evaluation environment.

### Correctness
More coming. We also want to highlight community effort such as [BackendBench](https://www.youtube.com/watch?v=BTfjdyZOKww).

### Performance
We highly recommend watching this [lecture](https://www.youtube.com/watch?v=1i7dxoAfKOU) from GPU mode on kernel profiling. 

We have (and continue to) implement various approaches to conduct kernel timing to understand the tradeoffs.

Check out `timing.py` to see available timing methods and `src/unit_tests/test_eval_timing.py` to test out various timing methods (including leveraging `cuda_event` marker, Triton `do_bench`, `host_time` E2E time). @palic and team is working on a blogpost explaining the different tradeoffs soon. 

### Profiling
We have experimental profiling support leveraging NVIDIA NCU in `profile.py`.

### Checkers
There are potentially many ways model might reward hack and we would like to catch the known ways through checkers [experimental and WIP]. We start with `kernel_static_checker.py`, which is a regex-based checker on the genenrated code against set of rules. We plan to add AST-based, LM-as-a-judge, and more runtime checks in the future. We welcome suggestions and contributions here.

### Unit Tests with Adversarial Examples
We've included some unit tests for the eval script in `src/unit_tests/test_eval_adversarial.py`. These tests run adversarial kernels (see `src/unit_tests/test_kernels/`) that contain examples of reward hacking that we've seen from LLMs and ensures that the eval script catches them, either by failing their correctness checks or flagging them for excessive speedups. Examples include:
- Reusing computations cached during the PyTorch reference
- Modifying inputs to cheat correctness checks
- Moving computation to a non-default CUDA stream

We will continue to add more tests as we explore additional adversarial scenarios.


Note: KernelBench is an ongoing open-source effort — please help us with issues and PRs!


Shoutout to @bkal01, @palic, @miru_why, @ngc92, @itsclivetime, for their suggestions and feedback. 