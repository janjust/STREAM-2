# STREAM
A modern implementation of the STREAM benchmarks.

Reference Implementation: http://www.cs.virginia.edu/stream/ref.html

Enhancements include:
 * Reporting performance for different vector sizes with a single build and run cycle
 * Auto scaling of number of iterations based on vector size
 * Add a new benchmark to report effective reduction bandwidth

## Benchmarks
Supports the standard Stream benchmarks. To calculate the memory bandwidth, the raw bandwidth is multiplied by the number of vectors (2 or 3) except for the new reduce benchmark. By default FP64 datatype (`double`) is used.

### Copy
Copy one vector to a second vector

`c[i] = a[i]`

### Scale
Scale vector by a scalar s and store in a second vector

`b[i] = s * c[i]`

### Add
Add two vectors and store in a third vector

`c[i] = a[i] + b[i]`

### Triad
Scale one vector by a scalar s, add to another, and store in third

`a[i] = b[i] + s * c[i]`

### Reduce
Add two vectors and store in the first vector. Does not multiply the raw bandwidth by number of vectors. Reports the effective reduction bandwidth.

`a[i] = a[i] + b[i]`

## Build

Edit Makefile to select appropriate compile options for the architecture.

`$ make`

## Run

Specify number of OpenMP threads and run the benchmark.

`$ OMP_NUM_THREADS=1 ./stream`
