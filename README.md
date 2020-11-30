# An Improved GPU-Accelerated Heuristic Technique Applied to the Capacitated Vehicle Routing Problem

This is the repository hosting the source code of the GPU implementation of the paper:
Marwan F. Abdelatti and Manbir S. Sodhi. 2020. An Improved GPU-Accelerated Heuristic Technique Applied to the Capacitated Vehicle Routing Problem. _In Proceedings of the Genetic and Evolutionary Computation Conference 2020 (GECCO ’20). ACM, New York, NY, USA, 9 pages._ You can access the paper from [this link.](https://dl.acm.org/doi/pdf/10.1145/3377930.3390159?casa_token=1svTNWgfQ-0AAAAA:Lwv63kPOpBMb40Wb7Pyn8YpnMYVgJLc7xycLJjpT_T0IXRQ9RLoOvnbNssZEqERN8beoM_FY-jB-)

In this work, an improved genetic algorithm is designed to be entirely executed on an NVIDIA GPU, taking advantage of the special CUDA GPU architecture to solving the CVRP. By distributing array elements over the GPU grid and using GPU kernel functions, the proposed algorithm successfully provides high-quality solutions within reasonable computational times, and near-optimal solutions for smaller benchmark problems. The algorithm entirely runs on an NVIDIA GPU.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

#### Hardware

To run the algorithm successfully, An NVIDIA GPU of Pascal or Turing architecture with 6+ GB of shared memory (e.g., NVIDIA RTX 2080) is required.
#### Software libraries

The following software and libraries are essintial to run the algorithm:
- Python 3.6 or higher.
- Proper NVIDIA GPU driver.
- NVIDIA Cuda 10.
- Numba
- Cupy.
- Numpy.

## Running the tests

To run the algorithm, open a command line instance then browse into the project directory. Use the following command:
```
python3 ga-vrp-gpu.py <problem-instance> <number-of-iterations (optional)>
```
You don't have to provide the full path of the problem file, just use the problem instance. For example:
```
user:~/GA_VRP_GPU$ python3 ga-vrp-gpu.py M-n151-k12 500000
```
The program will automatically read the problem file located at "GA_VRP_GPU/test_set/M/M-n151-k12.vrp"

## License

We will be glad if you find this code helpful for you. Use and improve it freely and give reference to this repository and the original research paper.
