# STREAM
A modern implementation of the STREAM benchmarks.

Reference Implementation: http://www.cs.virginia.edu/stream/ref.html

Enhancements include reporting performance for different vector sizes with a single build and run cycle.

## Benchmarks
Supports the standard Stream benchmarks. To calculate the memory bandwidth, the raw bandwidth is multiplied by the number of vectors (2 or 3) except for reduce.

### Copy
Copy one vector to another

`c[i] = a[i]`

### Scale
Scale vector by a scalar s

`b[i] = s * c[i]`

### Add
Add two vectors

`c[i] = a[i] + b[i]`

### Triad
Scale one vector by s and add to another

`a[i] = b[i] + s * c[i]`

### Reduce
Report the Reduction bandwidth of two vectors. Should be half of bandwidth reported by Add.

`a[i] = a[i] + b[i]`

## Build

Edit Makefile to select appropriate compile options for the architecture.

`$ make`

## Run

Specify number of OpenMP threads and run the benchmark.

`$ OMP_NUM_THREADS=1 ./stream`
