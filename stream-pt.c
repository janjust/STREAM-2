#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <pthread.h>
#include <sys/mman.h>
#include <linux/mman.h>

#define DTYPE double
DTYPE scalar = 1.234;

#define PEER    8
#define ROOT    0
#define WAIT    0
#define RUN     1

#define min(a,b) ((a) < (b) ? (a) : (b))

#define _256KB 262144

typedef struct thread_sync_t {
    volatile uint64_t v[16]; //128 bytes
} thread_sync_t;

typedef struct thread_ctx_t thread_ctx_t;

/* Define Stream Benchmarks */
typedef void (*stream_fn)(DTYPE *, DTYPE *, DTYPE *, size_t, thread_ctx_t *);

typedef struct stream_t {
    int threads;
    thread_sync_t *thread_sync;
    stream_fn fn;

    size_t mincount;
    size_t maxcount;
    size_t maxbytes;

    void* a;
    void* b;
    void* c;

    size_t count;
    int iters;
} stream_t;

typedef struct thread_ctx_t {
    pthread_t id;
    int idx;
    stream_t *stream;

    void *a;
    void *b;
    void *c;
    void *d;
} thread_ctx_t;

/* Utility Functions */
double gettimeus(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

void set_data(DTYPE *buf, size_t maxbytes)
{
    memset((void *)buf, 123, maxbytes);
}

int get_num_iters(size_t nbytes)
{
    int iters = 1e4;
    if (nbytes > 1e3)
        iters = 1e3;
    if (nbytes > 1e5)
        iters = 1e2;
    if (nbytes > 1e7)
        iters = 1e1;
    if (nbytes > 1e9)
        iters = 1e0;
    return iters * 1000; // * 1000;
}

/* Compute Functions Based on STREAM Benchmark */
void reduce_pthread(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict d,
    size_t count, thread_ctx_t *ctx)
{
    
    volatile DTYPE tmp;
    for (size_t i = 0; i < count; i++) {
        a[i] *= b[i];
    }

    /* Invalidate get data */
    for (size_t i = 0; i < count; i++) {
        tmp = d[i];
    }
}

typedef struct benchmark
{
    char name[20];
    int num_vectors;
    stream_fn fn;
} benchmark;

benchmark benchmarks[] = {
    {"reduce_pthread", 1, reduce_pthread},
};

void print_header(stream_t *stream)
{
    printf("Compile Flags: %s\n", OPTFLAGS);
    printf ("Threads = %d\n", stream->threads);
    printf("Element Size: %ld Bytes    ", sizeof(DTYPE));
    printf("Reported BW: GByte/s\n");
    printf("%-8s %-12s", "Iters", "Bytes");

    int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
    for (int i = 0; i < num_bench; i++)
    {
        printf("%15s", benchmarks[i].name);
    }
    printf("\n");
}

void release_peer_threads(thread_sync_t *sync, int threads)
{
    int i;

    for (i = 1; i < threads; i++) {
        sync[i].v[ROOT] = RUN;
        sync[i].v[PEER] = RUN;
    }

    return;
}

void wait_for_peer_threads(thread_sync_t *sync, int threads)
{
    int i;
    volatile int done = 0;
    do {
        done = threads-1;
        for (i = 1; i < threads; i++) {
            if (sync[i].v[ROOT] == WAIT) {
                done--;
            }
        }
    } while(done);

    return;
}

static void thread_set_affinity(thread_ctx_t *thread_ctx)
{
    int places = 8 / thread_ctx->stream->threads;
    pthread_t thread = pthread_self();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    int i=0, j=0;
    CPU_SET(thread_ctx->idx * places, &cpuset);

    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset); 
}

double stream_root_thread(void *arg)
{
    thread_ctx_t *thread_ctx = (thread_ctx_t *)arg;
    stream_t *stream = thread_ctx->stream;
    thread_sync_t *thread_sync = stream->thread_sync;

    size_t count = stream->count;
    size_t offset = thread_ctx->idx * count;

    double  t_start = 0.0, t_end = 0.0;

    DTYPE   *ta = thread_ctx->a,
            *tb = thread_ctx->b,
            *tc = thread_ctx->c,
            *td = thread_ctx->d;
    
    // DTYPE   *sa = stream->a,
    //         *sb = stream->b,
    //         *sc = stream->c;

    thread_set_affinity(thread_ctx);

    /* warm up */
    for (int i = 0; i < 10; i++) {
        release_peer_threads(thread_sync, stream->threads);
        // (stream->fn)(&sa[offset], &sb[offset], NULL, count, thread_ctx);
        (stream->fn)(ta, tb, td, count, thread_ctx);
        wait_for_peer_threads(thread_sync, stream->threads);
    }

    t_start = gettimeus();
    for (int i = 0; i < stream->iters; i++)
    {
        release_peer_threads(thread_sync, stream->threads);

        // (stream->fn)(&sa[offset], &sb[offset], NULL, count, thread_ctx);
        (stream->fn)(ta, tb, td, count, thread_ctx);
        wait_for_peer_threads(thread_sync, stream->threads);
    }
    t_end = gettimeus();

    return t_end - t_start;
}

void *stream_peer_thread(void *arg)
{
    thread_ctx_t *thread_ctx = (thread_ctx_t *)arg;
    stream_t *stream = thread_ctx->stream;
    thread_sync_t *sync = &stream->thread_sync[thread_ctx->idx];

    size_t count = stream->count;
    size_t offset = thread_ctx->idx * count;

    int iters = stream->iters;

    DTYPE   *ta = thread_ctx->a,
            *tb = thread_ctx->b,
            *tc = thread_ctx->c,
            *td = thread_ctx->d;

    // DTYPE   *sa = stream->a,
    //         *sb = stream->b,
    //         *sc = stream->c;

    thread_set_affinity(thread_ctx);

    /* warm up*/
    for (int j = 0; j < 10; j++) {
        while (sync->v[PEER] == WAIT) {
        }
        sync->v[PEER] = WAIT;

        // (stream->fn)(&sa[offset], &sb[offset], NULL, count, thread_ctx);
        (stream->fn)(ta, tb, td, count, thread_ctx);

        sync->v[ROOT] = WAIT;
    }

    // run all iterations
    for (int j = 0; j < iters; j++)
    {
        /*wait for root thread*/
        while (sync->v[PEER] == WAIT) {
        }
        sync->v[PEER] = WAIT;

        // (stream->fn)(&sa[offset], &sb[offset], NULL, count, thread_ctx);
        (stream->fn)(ta, tb, td, count, thread_ctx);

        /* Signal root */
        sync->v[ROOT] = WAIT;
    }
}

/* Report Raw Latency in Microsecond from Individual Benchmarks */
double run_bench(stream_fn fn, DTYPE *a, DTYPE *b, DTYPE *c, size_t count, int iters,
    thread_ctx_t *thread_ctx)
{
    stream_t *stream = thread_ctx[0].stream;
    stream->iters = iters;
    stream->count = count / stream->threads;
    stream->fn = fn;

    double lat_us = 0.0;
    for (int i = 1; i < stream->threads; i++) {
        pthread_create(&thread_ctx[i].id, NULL, stream_peer_thread, &thread_ctx[i]);
    }

    lat_us = stream_root_thread((void *)&thread_ctx[0]);

    for (size_t i = 1; i < stream->threads; i++) {
        pthread_join(thread_ctx[i].id, NULL);
    }

    return lat_us / iters;
}

int main(int argc, char **argv)
{
    thread_ctx_t *thread_ctx = NULL;
    char *env = NULL;
    int ret;

    stream_t *stream = calloc(1, sizeof(stream_t));

    env = getenv("STREAM_THREADS");
    stream->threads = (env == NULL) ? 1 : atoi(env);
    stream->thread_sync = calloc(stream->threads, sizeof(thread_sync_t));
    thread_ctx = calloc(stream->threads, sizeof(thread_ctx_t));
    size_t pg_size = sysconf(_SC_PAGESIZE);

    stream->mincount = (1 << 20) / sizeof(DTYPE);
    stream->maxcount = (1 << 20) / sizeof(DTYPE);
    stream->maxbytes = stream->maxcount * sizeof(DTYPE);

    // printf ("Done and sleeping\n");
    // sleep(30);

    size_t bytes = stream->maxbytes / stream->threads;

    printf ("bytes = %d\n", bytes);

    /* setup thread ctxs */
    for (int i = 0; i < stream->threads; i++) {
        thread_ctx[i].idx = i;
        thread_ctx[i].stream = stream;

        thread_ctx[i].a = mmap(NULL, 4 * bytes,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        thread_ctx[i].b = (char *)thread_ctx[i].a + bytes;
        thread_ctx[i].c = (char *)thread_ctx[i].b + bytes;
        thread_ctx[i].d = (char *)thread_ctx[i].c + bytes;
        // set_data(thread_ctx[i].a, 4 * bytes);
    }

    // stream->a = aligned_alloc(pg_size, stream->maxbytes);
    // stream->b = aligned_alloc(pg_size, stream->maxbytes);

    // set_data(stream->a, stream->maxbytes);
    // set_data(stream->b, bytes);
    
    print_header(stream);

    for (size_t count = stream->mincount; count <= stream->maxcount; count *= 2)
    {
        size_t nbytes = count * sizeof(DTYPE);
        double lat_us = 0.0;
        int iters = get_num_iters(nbytes);

        printf("%-8d %-12ld", iters, nbytes);

        int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
        for (int i = 0; i < num_bench; i++)
        {
            benchmark bench = benchmarks[i];
            lat_us = run_bench(bench.fn,
                NULL, NULL, NULL,
                count, iters, thread_ctx);
            double bw_GBs = 1e-3 * nbytes / lat_us;
            bw_GBs *= bench.num_vectors;

            printf("%15.2lf ", bw_GBs);
        }
        printf("\n");
    }

    for (int i = 0; i < stream->threads; i++) {
        munmap(thread_ctx[i].a, stream->maxbytes/stream->threads);
        munmap(thread_ctx[i].b, stream->maxbytes/stream->threads);
        // munmap(thread_ctx[i].c, stream->maxbytes/stream->threads);
    }

    // free(stream->a);
    // free(stream->b);

    return 0;
}
