#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#define DTYPE double
DTYPE scalar = 1.234;

/* Utility Functions */
double gettimeus(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

void set_data(DTYPE *buf, size_t maxbytes)
{
    memset((void*)buf, 123, maxbytes);
}

int get_omp_num_threads(void)
{
    int nt;
#pragma omp parallel
    {
#pragma omp master
    {
        nt = omp_get_num_threads();
    }
    }
    return nt;
}

int get_num_iters(size_t nbytes)
{
    int iters = 1e4;
    if (nbytes > 1e3) iters = 1e3;
    if (nbytes > 1e5) iters = 1e2;
    if (nbytes > 1e7) iters = 1e1;
    if (nbytes > 1e9) iters = 1e0;
    return iters;
}

/* Compute Functions Based on STREAM Benchmark */
void stream_copy(DTYPE* restrict a, DTYPE* restrict b, DTYPE* restrict c, size_t count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i];
    }
}

void stream_scale(DTYPE* restrict a, DTYPE* restrict b, DTYPE* restrict c, size_t count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        b[i] = scalar * c[i];
    }
}

void stream_add(DTYPE* restrict a, DTYPE* restrict b, DTYPE* restrict c, size_t count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        c[i] = a[i] + b[i];
    }
}

void stream_triad(DTYPE* restrict a, DTYPE* restrict b, DTYPE* restrict c, size_t count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

void stream_reduce(DTYPE* restrict a, DTYPE* restrict b, DTYPE* restrict c, size_t count)
{
    #pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
        a[i] = a[i] + b[i];
    }
}

/* Define Stream Benchmarks */
typedef void (*stream_fn)(DTYPE*, DTYPE*, DTYPE*, size_t);

typedef struct benchmark {
    char name[10];
    int  num_vectors;
    stream_fn fn;
} benchmark;

benchmark benchmarks[] = {
    { "copy",   2, stream_copy   },
    { "scale",  2, stream_scale  },
    { "add",    3, stream_add    },
    { "triad",  3, stream_triad  },
    { "reduce", 1, stream_reduce },
};

/* Report Raw Latency in Microsecond from Individual Benchmarks */
double run_bench(stream_fn fn, DTYPE *a, DTYPE *b, DTYPE *c, size_t count, int iters)
{
    /* Ignore first iteration */
    (fn)(a, b, c, count);

    double t_start = gettimeus();
    for (int j = 0; j < iters; j++) {
        (fn)(a, b, c, count);
    }
    double t_end = gettimeus();

    double lat_us = (t_end - t_start) / iters;
    return lat_us;
}

void print_header()
{
    printf("Compile Flags: %s\n", OPTFLAGS);
    printf("Element Size: %ld Bytes    ", sizeof(DTYPE));
    printf("OpenMP Threads: %d    ", get_omp_num_threads());
    printf("Reported BW: GByte/s\n");
    printf("%-8s %-12s", "Iters", "Bytes");

    int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
    for (int i = 0; i < num_bench; i++) {
        printf("%10s", benchmarks[i].name);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    size_t pg_size  = sysconf(_SC_PAGESIZE);
    size_t mincount = (1<<12)  / sizeof(DTYPE);
    size_t maxcount = (1<<30)  / sizeof(DTYPE);
    size_t maxbytes = maxcount * sizeof(DTYPE);

    DTYPE *a = aligned_alloc(pg_size, maxbytes);
    DTYPE *b = aligned_alloc(pg_size, maxbytes);
    DTYPE *c = aligned_alloc(pg_size, maxbytes);

    set_data(a, maxbytes);
    set_data(b, maxbytes);
    set_data(c, maxbytes);

    print_header();

    for (size_t count = mincount; count <= maxcount; count *= 2) {
        size_t nbytes = count * sizeof(DTYPE);
        int iters = get_num_iters(nbytes);
        printf("%-8d %-12ld", iters, nbytes);

        int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
        for (int i = 0; i < num_bench; i++) {
            benchmark bench = benchmarks[i];
            double lat_us = run_bench(bench.fn, a, b, c, count, iters);
            double bw_GBs = 1e-3 * nbytes / lat_us;
            bw_GBs *= bench.num_vectors;
            printf("%10.2lf", bw_GBs);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}

