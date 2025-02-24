#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>
#include <assert.h>
#include <pthread.h>

#define MAX_RKEY_LEN 1024
#define DTYPE double
DTYPE scalar = 1.234;

#define PEER    8
#define ROOT    0
#define WAIT    0
#define RUN     1

typedef struct thread_ctx_t thread_ctx_t;

/* Define Stream Benchmarks */
typedef void (*stream_fn)(DTYPE *, DTYPE *, DTYPE *, size_t, thread_ctx_t *);

typedef struct thread_sync_t {
    volatile uint64_t v[16]; //128 bytes
} thread_sync_t;

typedef struct stream_mem_rkey_t
{
    void *rkey_buf;
    size_t rkey_buf_len;
} stream_mem_rkey_t;

typedef struct stream_mem_t
{
    void *base;
    ucp_mem_h memh;
    stream_mem_rkey_t rkey;
    void *rem_base;
    ucp_rkey_h rem_rkey;
} stream_mem_t;

typedef struct _stream_ucx_t
{
    /* Local UCX stuff */
    ucp_context_h ucp_ctx;
    ucp_worker_h ucp_worker;
    ucp_worker_attr_t worker_attr;
    ucp_request_param_t req_param;

    /* Remote UCX stuff */
    void *rem_worker_addr;
    size_t rem_worker_addr_len;
    ucp_ep_h remote_ep;

    stream_fn fn;
    stream_mem_t buffer_a;
    stream_mem_t buffer_b;
    stream_mem_t buffer_c;

    thread_sync_t *thread_sync;
    int threads;
    int rank;
    int size;
    
    size_t mincount;
    size_t maxcount;
    size_t maxbytes;
    int iters;
    size_t count;
    double t_get_end;

    char rem_rkey_buf[MAX_RKEY_LEN];
} stream_ucx_t;

typedef struct thread_ctx_t {
    int idx;
    pthread_t id;
    stream_ucx_t *stream_net;
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
    if (nbytes > 1e3)
        iters = 1e3;
    if (nbytes > 1e5)
        iters = 1e2;
    if (nbytes > 1e7)
        iters = 1e1;
    if (nbytes > 1e9)
        iters = 1e0;
    return iters * 10;
}

/* Compute Functions Based on STREAM Benchmark */
void stream_copy(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count,thread_ctx_t *ctx)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        c[i] = a[i];
    }
}

void stream_scale(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count,thread_ctx_t *ctx)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        b[i] = scalar * c[i];
    }
}

void stream_add(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count,thread_ctx_t *ctx)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void stream_triad(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count,thread_ctx_t *ctx)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; i++)
    {
        a[i] = b[i] + scalar * c[i];
    }
}

void stream_reduce(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count,thread_ctx_t *ctx)
{
#pragma omp parallel for
        for (size_t i = 0; i < count; i++)
        {
            a[i] *= b[i];
        }
}

void stream_reduce_get(DTYPE *restrict a, DTYPE *restrict b, DTYPE *restrict c,
    size_t count, thread_ctx_t *ctx)
{
    for (size_t i = 0; i < count; i++) {
        a[i] *= b[i];
    }
}

typedef struct benchmark
{
    char name[10];
    int num_vectors;
    stream_fn fn;
} benchmark;

benchmark benchmarks[] = {
    {"copy", 2, stream_copy},
    {"scale", 2, stream_scale},
    {"add", 3, stream_add},
    {"triad", 3, stream_triad},
    {"reduce", 1, stream_reduce},
    {"get-reduce", 1, stream_reduce_get},
};

ucs_status_t ucx_request_wait(ucp_worker_h ucp_worker, ucs_status_ptr_t request)
{
    ucs_status_t status;

    /* immediate completion */
    if (request == NULL)
    {
        return UCS_OK;
    }
    else if (UCS_PTR_IS_ERR(request))
    {
        status = ucp_request_check_status(request);
        fprintf(stderr, "unable to complete UCX request (%s)\n", ucs_status_string(status));
        return UCS_PTR_STATUS(request);
    }
    else
    {
        do
        {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
    }

    return status;
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

static void thread_set_affinity(thread_ctx_t *ctx)
{
    int places = 8/ctx->stream_net->threads;
    pthread_t thread = pthread_self();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    CPU_SET(ctx->idx * places, &cpuset);

    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset); 
}

double stream_root_thread(void *arg) {
    ucs_status_t status;
    ucs_status_ptr_t ucp_req = UCS_OK;
    ucp_request_param_t req_param = {};

    thread_ctx_t *thread_ctx = (thread_ctx_t *)arg;
    stream_ucx_t *stream_net = thread_ctx->stream_net;
    thread_sync_t *thread_sync = stream_net->thread_sync;

    thread_set_affinity(thread_ctx);

    DTYPE   *a = stream_net->buffer_a.base,
            *b = stream_net->buffer_b.base,
            *c = stream_net->buffer_c.base;

    int tidx = thread_ctx->idx;
    int iters = stream_net->iters;
    int j;
    size_t count = stream_net->count / stream_net->threads;

    double t_get_start = 0.0, t_get_end = 0.0;
    stream_net->t_get_end = 0.0;

    DTYPE *tmp_get = aligned_alloc(sysconf(_SC_PAGESIZE), stream_net->maxbytes),
          *tmp_acc = aligned_alloc(sysconf(_SC_PAGESIZE), stream_net->maxbytes);

    /* warm up */
    for (j = 0; j < 10; j++) {
        release_peer_threads(thread_sync, stream_net->threads);
        ucp_req = ucp_get_nbx(stream_net->remote_ep, b, stream_net->count * sizeof(DTYPE),
                (uint64_t)stream_net->buffer_b.rem_base,
                stream_net->buffer_b.rem_rkey, &req_param);
        status = ucx_request_wait(stream_net->ucp_worker, ucp_req);
        wait_for_peer_threads(thread_sync, stream_net->threads);
    }

    double t_start = gettimeus();
    for (j = 0; j < iters; j++)
    {
        /* Get the data */
        #if 1
        t_get_start = gettimeus();
        ucp_req = ucp_get_nbx(stream_net->remote_ep, b, stream_net->count * sizeof(DTYPE),
                        (uint64_t)stream_net->buffer_b.rem_base,
                        stream_net->buffer_b.rem_rkey, &req_param);
        status = ucx_request_wait(stream_net->ucp_worker, ucp_req);
        t_get_end += gettimeus() - t_get_start;        
        #endif
        /* ---- */

        release_peer_threads(thread_sync, stream_net->threads);

        (stream_net->fn)(a, b, c, count, thread_ctx);

        wait_for_peer_threads(thread_sync, stream_net->threads);
    }
    double t_end = gettimeus();
    free(tmp_get);
    free(tmp_acc);
    return t_end - t_start;// - t_get_end;
}

void *stream_peer_thread(void *arg) {
    thread_ctx_t *thread_ctx = (thread_ctx_t *)arg;
    stream_ucx_t *stream_net = thread_ctx->stream_net;
    thread_sync_t *sync = stream_net->thread_sync;

    int iters = stream_net->iters;
    size_t count = stream_net->count / stream_net->threads;
    //size_t count_offset = stream_net->maxcount / stream_net->threads; // ->idx * count;
    size_t count_offset = thread_ctx->idx * count;
    // printf ("thread [%d], count = %d at offset%d\n", thread_ctx->idx, count, count_offset);

    thread_set_affinity(thread_ctx);

    DTYPE   *a = stream_net->buffer_a.base,
            *b = stream_net->buffer_b.base,
            *c = stream_net->buffer_c.base;

    DTYPE *tmp_get = aligned_alloc(sysconf(_SC_PAGESIZE), stream_net->maxbytes),
          *tmp_acc = aligned_alloc(sysconf(_SC_PAGESIZE), stream_net->maxbytes);

    /* warm up*/
    for (int j = 0; j < 10; j++) {
        while (stream_net->thread_sync[thread_ctx->idx].v[PEER] == WAIT) {
        }
        stream_net->thread_sync[thread_ctx->idx].v[PEER] = WAIT;
        stream_net->thread_sync[thread_ctx->idx].v[ROOT] = WAIT;
    }

    // compute
    for (int j = 0; j < iters; j++)
    {
        /*wait for root thread*/
        while (stream_net->thread_sync[thread_ctx->idx].v[PEER] == WAIT) {
        }
        stream_net->thread_sync[thread_ctx->idx].v[PEER] = WAIT;

         (stream_net->fn)(&a[count_offset], &b[count_offset], &c[count_offset],
             count, thread_ctx);

        stream_net->thread_sync[thread_ctx->idx].v[ROOT] = WAIT;
    }
    
    free(tmp_get);
    free(tmp_acc);
}

/* Report Raw Latency in Microsecond from Individual Benchmarks */
double run_bench(stream_fn fn, DTYPE *a, DTYPE *b, DTYPE *c, size_t count, int iters,
    thread_ctx_t *thread_ctx)
{
    double t_end = 0.0, t_start = 0.0;
    stream_ucx_t *stream_net = thread_ctx->stream_net;
    stream_net->count = count;
    stream_net->iters = iters;
    stream_net->fn = fn;

    // printf ("count = %d\n", count);

    if (stream_net->rank > 0) {
        return 0.0;
    }

    // sleep(20);
    
    /* Ignore first iteration */
    if (thread_ctx->idx == 0) {
        (fn)(a, b, c, count, thread_ctx);
    }

    double lat_us = 0.0;
    if (fn == stream_reduce_get) {

        for (int i = 1; i < stream_net->threads; i++) {
            pthread_create(&thread_ctx[i].id, NULL, stream_peer_thread, &thread_ctx[i]);
        }

        // t_start = gettimeus();
        lat_us = stream_root_thread((void *)&thread_ctx[0]);
        // t_end = gettimeus();// - stream_net->t_get_end;

        for (size_t i = 1; i < stream_net->threads; i++) {
            pthread_join(thread_ctx[i].id, NULL);
        }
    }
    else {
        // t_start = gettimeus();
        for (int j = 0; j < iters; j++)
        {
            (fn)(a, b, c, count, thread_ctx);
        }
        // t_end = gettimeus();
    }

    // double lat_us = (t_end - t_start) / iters;
    return lat_us / iters;
}

void print_header()
{
    printf("Compile Flags: %s\n", OPTFLAGS);
    printf("Element Size: %ld Bytes    ", sizeof(DTYPE));
    // printf("OpenMP Threads: %d    ", get_omp_num_threads());
    printf("Reported BW: GByte/s\n");
    printf("%-8s %-12s", "Iters", "Bytes");

    int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
    for (int i = num_bench-1; i < num_bench; i++)
    {
        printf("%11s", benchmarks[i].name);
    }
    printf("\n");
}
static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
}

static int stream_ucx_init(stream_ucx_t *stream_net)
{
    ucp_params_t ucp_params;
    ucs_status_t status;
    ucp_worker_params_t worker_params;
    int ret = UCS_OK;

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG |
                          UCP_FEATURE_RMA;

    ucp_config_t *ucp_config;
    ucp_config_read(NULL, NULL, &ucp_config);

    status = ucp_init(&ucp_params, ucp_config, &stream_net->ucp_ctx);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_init(%s)\n", ucs_status_string(status));
        ret = UCS_ERR_NO_MESSAGE;
        goto err;
    }

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create (
            stream_net->ucp_ctx, &worker_params, &stream_net->ucp_worker);
    if (status != UCS_OK)
    {
        fprintf(stderr, "failed to create ucp_worker (%s)\n", ucs_status_string(status));
        ret = UCS_ERR_NO_MESSAGE;
        goto err_cleanup;
    }

    stream_net->worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                                        UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    stream_net->worker_attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    status = ucp_worker_query(stream_net->ucp_worker, &stream_net->worker_attr);
    if (UCS_OK != status) {
        fprintf(stderr, "failed to ucp_worker_query (%s)\n", ucs_status_string(status));
        ret = UCS_ERR_NO_MESSAGE;
        goto err_worker;
    }

    return ret;
err_worker:

    ucp_worker_destroy(stream_net->ucp_worker);

err_cleanup:
    ucp_cleanup(stream_net->ucp_ctx);
err:
    return ret;
}


static int stream_connect_workers(stream_ucx_t *stream_net)
{
    MPI_Status mpi_status;

    MPI_Sendrecv(
        &stream_net->worker_attr.address_length, sizeof(size_t), MPI_BYTE, !stream_net->rank, 0,
        &stream_net->rem_worker_addr_len, sizeof(size_t), MPI_BYTE, !stream_net->rank, 0,
        MPI_COMM_WORLD, &mpi_status);

    stream_net->rem_worker_addr = calloc(1, stream_net->rem_worker_addr_len);

    MPI_Sendrecv(
        stream_net->worker_attr.address, stream_net->worker_attr.address_length, MPI_BYTE,
            !stream_net->rank, 0, 
            stream_net->rem_worker_addr, stream_net->rem_worker_addr_len, MPI_BYTE,
            !stream_net->rank, 0, MPI_COMM_WORLD, &mpi_status);

    ucs_status_t status = UCS_OK;

    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = err_cb;
    ep_params.address = stream_net->rem_worker_addr;

    status = ucp_ep_create(stream_net->ucp_worker, &ep_params, &stream_net->remote_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "!!! failed to create endpoint to remote %d (%s)\n",
                status, ucs_status_string(status));
    }

    free(stream_net->rem_worker_addr);
    return 0;
}

static void stream_buffer_free(stream_ucx_t *stream_net, stream_mem_t *mem)
{
    ucp_rkey_buffer_release(mem->rkey.rkey_buf);
    ucp_rkey_destroy(mem->rem_rkey);
    ucp_mem_unmap(stream_net->ucp_ctx, mem->memh);
    mem->base = NULL;
}

static int exchange_rkeys(stream_ucx_t *stream_net, stream_mem_t *buffer)
{
    MPI_Status mpi_status;
    ucs_status_t status;
    char tmp_rkey_loc[MAX_RKEY_LEN],
         tmp_rkey_rem[MAX_RKEY_LEN];
    
    uint64_t tmp_src = (uint64_t)buffer->base,
             tmp_dst = 0;

    MPI_Sendrecv(
        &tmp_src, sizeof(uint64_t), MPI_BYTE, !stream_net->rank, 0,
        &tmp_dst, sizeof(uint64_t), MPI_BYTE, !stream_net->rank, 0,
        MPI_COMM_WORLD, &mpi_status);

    buffer->rem_base = (uint64_t *) tmp_dst;

    memcpy(&tmp_rkey_loc, buffer->rkey.rkey_buf,
           buffer->rkey.rkey_buf_len);

    MPI_Sendrecv(&tmp_rkey_loc, MAX_RKEY_LEN, MPI_BYTE, !stream_net->rank, 0,
                 &tmp_rkey_rem, MAX_RKEY_LEN, MPI_BYTE, !stream_net->rank, 0,
                 MPI_COMM_WORLD, &mpi_status);

    status = ucp_ep_rkey_unpack(stream_net->remote_ep,
                (void *)tmp_rkey_rem, &buffer->rem_rkey);
    if (status != UCS_OK)
    {
        fprintf(stderr, "unable to unpack key!\n");
        return 1;
    }

    return 0;
}

static void stream_register_buffer(stream_ucx_t *stream_net, DTYPE *buffer,
    stream_mem_t *mem, size_t size)
{
    ucp_mem_attr_t mem_attr;
    ucs_status_t status;

    memset(mem, 0, sizeof(*mem));
    mem->base = buffer;

    ucp_mem_map_params_t mem_params = {
        .address = mem->base,
        .length  = size,
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                     UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                     UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
    };

    status = ucp_mem_map(stream_net->ucp_ctx, &mem_params, &mem->memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_map (%s)\n", ucs_status_string(status));
        return;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;

    status = ucp_mem_query(mem->memh, &mem_attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_mem_query (%s)\n", ucs_status_string(status));
        ucp_mem_unmap(stream_net->ucp_ctx, mem->memh);
        return;
    }

    assert(mem_attr.length >= mem_params.length);
    assert(mem_attr.address <= mem_params.address);

    status = ucp_rkey_pack(stream_net->ucp_ctx, mem->memh,
                           &mem->rkey.rkey_buf,
                           &mem->rkey.rkey_buf_len);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status));
        ucp_mem_unmap(stream_net->ucp_ctx, mem->memh);
        return;
    }
    assert(mem->rkey.rkey_buf_len < MAX_RKEY_LEN);

    return;
}

static int stream_ucx_fini(stream_ucx_t *stream_net)
{
    ucp_worker_release_address(stream_net->ucp_worker, stream_net->worker_attr.address);
    ucp_worker_destroy(stream_net->ucp_worker);
    ucp_cleanup(stream_net->ucp_ctx);
}

int main(int argc, char **argv)
{
    ucs_status_t status;
    ucs_status_ptr_t ucp_req = UCS_OK;

    char *env = NULL;

    int rank, ret;

    ucp_request_param_t req_param = {};
    memset(&req_param, 0, sizeof(ucp_request_param_t));

    stream_ucx_t *stream_net = calloc(1, sizeof(stream_ucx_t));
    env = getenv("STREAM_THREADS");
    stream_net->threads = (env == NULL) ? 0 : atoi(env);
    stream_net->thread_sync = calloc(stream_net->threads, sizeof(thread_sync_t));
    stream_net->buffer_a.rem_rkey = calloc(stream_net->threads, sizeof(ucp_rkey_h));
    stream_net->buffer_b.rem_rkey = calloc(stream_net->threads, sizeof(ucp_rkey_h));
    stream_net->buffer_c.rem_rkey = calloc(stream_net->threads, sizeof(ucp_rkey_h));

    thread_ctx_t *thread_ctx = calloc(stream_net->threads, sizeof(thread_ctx_t));
    for (int i = 0; i < stream_net->threads; i++) {
        thread_ctx[i].stream_net = stream_net;
        thread_ctx[i].idx = i;
    }

    // sleep(20);

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &stream_net->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &stream_net->size);

    rank = stream_net->rank;
	char hostname[1024];
	gethostname(hostname, 1024);

    printf(" Rank %d/%d, pid=%d, %s\n", stream_net->rank, stream_net->size, getpid(), hostname);

    size_t pg_size = sysconf(_SC_PAGESIZE);
    stream_net->mincount = (1 << 12) / sizeof(DTYPE);
    stream_net->maxcount = (1 << 22) / sizeof(DTYPE);
    stream_net->maxbytes = stream_net->maxcount * sizeof(DTYPE);

    DTYPE *a = aligned_alloc(pg_size, stream_net->maxbytes);
    DTYPE *b = aligned_alloc(pg_size, stream_net->maxbytes);
    DTYPE *c = aligned_alloc(pg_size, stream_net->maxbytes);

    set_data(a, stream_net->maxbytes);
    set_data(b, stream_net->maxbytes);
    set_data(c, stream_net->maxbytes);

    if (UCS_OK != stream_ucx_init(stream_net))
    {
        fprintf(stderr, "Error in UCX init!\n");
    }
    stream_connect_workers(stream_net);
    stream_register_buffer(stream_net, a, &stream_net->buffer_a, stream_net->maxbytes);
    stream_register_buffer(stream_net, b, &stream_net->buffer_b, stream_net->maxbytes);
    stream_register_buffer(stream_net, c, &stream_net->buffer_c, stream_net->maxbytes);

    exchange_rkeys(stream_net, &stream_net->buffer_a);
    exchange_rkeys(stream_net, &stream_net->buffer_b);
    exchange_rkeys(stream_net, &stream_net->buffer_c);

    memset(&req_param, 0, sizeof(ucp_request_param_t));
    ucp_req = ucp_worker_flush_nbx(stream_net->ucp_worker, &req_param);
    status = ucx_request_wait(stream_net->ucp_worker, ucp_req);        

    MPI_Request mpi_request;
    MPI_Status mpi_status;
    int mpi_flag = 1;

    if (rank == 0)
        print_header();

    for (size_t count = stream_net->mincount; count <= stream_net->maxcount; count *= 2)
    {
        size_t nbytes = count * sizeof(DTYPE);
        double lat_us = 0.0;
        int iters = get_num_iters(nbytes);

        if (stream_net->rank == 0)
            printf("%-8d %-12ld", iters, nbytes);

        int num_bench = sizeof(benchmarks) / sizeof(benchmarks[0]);
        // for (int i = 0; i < num_bench; i++)
        for (int i = num_bench-1; i < num_bench; i++)
        {
            benchmark bench = benchmarks[i];
            lat_us = run_bench(bench.fn,
                stream_net->buffer_a.base, stream_net->buffer_b.base, stream_net->buffer_c.base,
                count, iters, thread_ctx);
            double bw_GBs = 1e-3 * nbytes / lat_us;
            bw_GBs *= bench.num_vectors;
            if (stream_net->rank == 0)
                printf("%11.2lf", bw_GBs);
        }
        if (stream_net->rank == 0)
            printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    stream_buffer_free(stream_net, &stream_net->buffer_a);
    stream_buffer_free(stream_net, &stream_net->buffer_b);
    stream_buffer_free(stream_net, &stream_net->buffer_c);

    stream_ucx_fini(stream_net);

    free(a);
    free(b);
    free(c);

    MPI_Finalize();
    return 0;
}
