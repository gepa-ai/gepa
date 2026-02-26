# CUDA Best Practices
The performance guidelines and best practices described in the CUDA C++ Programming Guide and
the CUDA C++ Best Practices Guide apply to all CUDA-capable GPU architectures. Programmers must
primarily focus on following those recommendations to achieve the best performance.
The high-priority recommendations from those guides are as follows:
- Find ways to parallelize sequential code,
- Minimize data transfers between the host and the device,
- Adjust kernel launch configuration to maximize device utilization,
- Ensure global memory accesses are coalesced,
- Minimize redundant accesses to global memory whenever possible,
- Avoid long sequences of diverged execution by threads within the same warp.