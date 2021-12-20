#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* Define Stream Operations */
typedef enum stream_op {
    COPY,
    SCALE,
    ADD,
    TRIAD,
    REDUCE,
    NUM_OPS
} stream_op;

char op_names[NUM_OPS][10] = {
    "copy",
    "scale",
    "add",
    "triad",
    "reduce",
};

int words_per_op[NUM_OPS] = {
    2,
    2,
    3,
    3,
    1,
};

typedef void (*stream_fn)(DTYPE*, DTYPE*, DTYPE*, size_t);
stream_fn stream_fns[NUM_OPS] = {
    stream_copy,
    stream_scale,
    stream_add,
    stream_triad,
    stream_reduce,
};

/* Report Raw Latency in Microsecond from Individual Benchmarks */
double bench(int op, DTYPE *a, DTYPE *b, DTYPE *c, size_t count, int iters)
{
    /* Ignore first iteration */
    (stream_fns[op])(a, b, c, count);

    double t_start = gettimeus();
    for (int j = 0; j < iters; j++) {
        (stream_fns[op])(a, b, c, count);
    }
    double t_end = gettimeus();

    double lat_us = (t_end - t_start) / iters;
    return lat_us;
}

void print_header()
{
    printf("Compile Flags: %s\n", OPTFLAGS);
    printf("Element Size: %-3ld ", sizeof(DTYPE));
    printf("OpenMP Threads: %-3d\n", get_omp_num_threads());

    printf("%-8s %-12s", "Iters", "Bytes");
    for (int op = 0; op < NUM_OPS; op++) {
        printf("%10s", op_names[op]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    size_t pg_size  = 4096;
    size_t mincount = pg_size / sizeof(DTYPE);
    size_t maxcount = 128l * 1024 * mincount;
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

        for (int op = 0; op < NUM_OPS; op++) {
            double lat_us = bench(op, a, b, c, count, iters);
            double bw_GBs = 1e-3 * nbytes / lat_us;
            bw_GBs *= words_per_op[op];
            printf("%10.2lf", bw_GBs);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
