# Parallel BiCGSTAB (MPI) for a Geodetic Boundary Value Problem

A hand-written **BiCGSTAB** iterative solver in C++ with an **MPI** parallel version, applied to a dense boundary-type collocation problem from physical geodesy: recovering the gravitational potential on the Earth's surface from gravity data — at full scale on **160,002 observation points**.

## Problem

- Observation points given by latitude, longitude and height, with gravity-disturbance data as Neumann boundary condition.
- Point sources are placed below the surface (method of fundamental solutions, a boundary-type method).
- Collocation leads to a **dense linear system** `G_N · α = q`, where `G_N` is the normal derivative of the fundamental solution `1 / (4π r)`.
- After solving for the source strengths `α`, the potential is evaluated as `u = G · α` and written to `solution_u.dat`.

## Problem sizes

| Run | Points `N` | Source depth `D` | Dense matrix | Where |
|-----|-----------:|-----------------:|--------------|-------|
| Development | 902 | 500 km | 902 × 902 | testing |
| Full | **160,002** | **50 km** | **160,002 × 160,002 (~205 GB in double precision)** | university cluster, 36 MPI processes |

The full-size matrix is far too large for a typical workstation, which is why each process assembles and stores only its own block of rows (~5.7 GB per process with 36 processes). The full run converged.

Constants are set with `#define` at the top of `main.cpp` (`N`, `D`, plus `GM`, `R`, `PI`); the repository version uses the development values.

## Parallelisation

- **Row-block decomposition:** every MPI process assembles only its own block of rows of `G_N`, so the full matrix is never stored on one process.
- **Distributed matrix–vector product:** each process multiplies its rows locally; the full vector is assembled with `MPI_Allgatherv`.
- **Global dot products and norms:** local partial sums combined with `MPI_Allreduce`.
- Input data are read on rank 0 and distributed with `MPI_Bcast`; the final potential is collected with `MPI_Gatherv`.

A serial `solveWithBiCGSTAB` is kept alongside the MPI version for reference.

## Build and run

**Windows — Visual Studio + Microsoft MPI** (how the project was developed)

1. Install Microsoft MPI (runtime and SDK) and add its `Include` directory and `msmpi.lib` to the project settings.
2. Build the project in Visual Studio.
3. Run from the project folder, with `BL-902.dat` in the working directory:

```powershell
mpiexec -n 2 .\Debug\MPIExample.exe
```

**Linux / HPC cluster**

```bash
mpicxx -O2 -std=c++17 main.cpp -o bem_mpi
mpirun -np 36 ./bem_mpi
```

Input format (`BL-902.dat`, one line per point): `B  L  H  q  <unused>` — latitude and longitude in degrees, height in metres, gravity data `q`.

## Example run (included 902-point dataset)

- BiCGSTAB converges in **10 iterations** from a residual norm of 1.1 × 10⁻² to 9.7 × 10⁻⁸ (tolerance 10⁻⁷).
- Runs on 1 and 4 processes produce **identical** results.
- The potential at all 902 points is written to `solution_u.dat` (`B  L  u`).
