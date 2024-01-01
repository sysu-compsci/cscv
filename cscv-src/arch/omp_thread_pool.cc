#include "omp_thread_pool.hpp"

#include <omp.h>

#include "arch/naive_numa_util.hpp"

using namespace std;

void OMP_thread_pool::execution_loop(int thread_id) {
    // PRINTF("Hello world from thread %d\n", thread_id);

    Execution_loop_param_pack given_param;

    while (true) {
        m_job_queues[thread_id].pop(given_param);
        // PRINTF("POPED %d in thread %d\n", given_param.m_type, thread_id);

        if (given_param.m_type == Execution_loop_param_pack::WITHOUT_THREAD_INFO) {
            Op_func op = force_cast_pointer<Op_func>(given_param.m_func_ptr);
            op(given_param.m_func_param);
        } else if (given_param.m_type == Execution_loop_param_pack::TERMINATE) {
            break;
        } else {
            ASSERT_AND_PRINTF(false, "Unknown type: %d\n", given_param.m_type);
        }

        m_ret_queues[thread_id].push(Thread_return_value(thread_id));
    }
}

void OMP_thread_pool::main_execution_loop() {
    setenv("OMP_STACKSIZE", strprintf("%luB", m_stack_size).c_str(), 1);
    printf("[OMP_thread_pool] Passed m_stack_size = %lu, get env OMP_STACKSIZE as %s\n",
        m_stack_size, getenv("OMP_STACKSIZE"));
    #pragma omp parallel num_threads(m_nthreads)
    {
        Naive_NUMA_util::get_instance().print_bound_info_in_omp_parallel_region();
        execution_loop(omp_get_thread_num());
    }
}

void OMP_thread_pool::join() {
    if (m_active) {
        for (int i = 0; i < m_nthreads; i++) {
            m_job_queues[i].push(Execution_loop_param_pack(Execution_loop_param_pack::TERMINATE, NULL, NULL));
        }
        pthread_join(m_main_thread, NULL);
    }

    m_active = false;
}

extern void kmp_set_stacksize(int size);
extern void kmp_set_stacksize_s(size_t size);

size_t parse_omp_stacksize_env() {
    char* omp_stack_size_ptr = getenv("OMP_STACKSIZE");

    if (omp_stack_size_ptr == nullptr) {
        return 0;
    }

    string omp_stack_size_str(omp_stack_size_ptr);

    size_t unit = 1;

    if (omp_stack_size_str.back() == 'B') {
        unit = 1;
    } else if (omp_stack_size_str.back() == 'K') {
        unit = 1024;
    } else if (omp_stack_size_str.back() == 'M') {
        unit = 1024 * 1024;
    } else if (omp_stack_size_str.back() == 'G') {
        unit = 1024 * 1024 * 1024;
    } else if (omp_stack_size_str.back() == 'T') {
        unit = 1024UL * 1024UL * 1024UL * 1024UL;
    }

    size_t base = 0, position = 0;
    while (position < omp_stack_size_str.size() && omp_stack_size_str[position] <= '9' && omp_stack_size_str[position] >= '0') {
        base *= 10;
        base += omp_stack_size_str[position] - '0';
        position++;
    }

    return base * unit;
}

void set_stack_size_and_check_pthread_attr(pthread_attr_t* thread_attr, size_t stack_size) {
    int pthread_ret;
    pthread_ret = pthread_attr_setstacksize(thread_attr, stack_size);
    ASSERT_AND_PRINTF(pthread_ret == 0, "");
    size_t new_stack_size;
    pthread_ret = pthread_attr_getstacksize(thread_attr, &new_stack_size);
    ASSERT_AND_PRINTF(pthread_ret == 0 && stack_size == new_stack_size, "");
}

OMP_thread_pool::OMP_thread_pool(int nthreads, int stack_size_mb) {
    m_job_queues.resize(nthreads);
    m_ret_queues.resize(nthreads);

    m_nthreads = nthreads;

    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    int pthread_ret;

    if (stack_size_mb > 0) {
        setenv("OMP_STACKSIZE", strprintf("%dM", stack_size_mb).c_str(), 1);
    }

    if (getenv("OMP_STACKSIZE") != nullptr) {
        size_t stack_size = parse_omp_stacksize_env();
        set_stack_size_and_check_pthread_attr(&thread_attr, stack_size);

        printf("[OMP_thread_pool] Passed stack_size_mb = %d, get env OMP_STACKSIZE as %s, parsed OMP_STACKSIZE as %lu bytes (take care about stack size in ulimit)\n",
            stack_size_mb, getenv("OMP_STACKSIZE"), stack_size);
        m_stack_size = stack_size;
    } else {
        printf("[OMP_thread_pool] Spawn omp thread with default stack size\n");
        // without OMP_STACKSIZE, the stack size should be set to OMP default
        if (sizeof(size_t) == 4) {
            m_stack_size = 2 * 1024 * 1024;
        } else {
            m_stack_size = 4 * 1024 * 1024;
        }
        set_stack_size_and_check_pthread_attr(&thread_attr, m_stack_size);
    }

    pthread_create(&m_main_thread, &thread_attr, force_cast_pointer<Pthread_func>(&OMP_thread_pool::main_execution_loop), this);

    m_active = true;
}

OMP_thread_pool::~OMP_thread_pool() {
    join();
}

void OMP_thread_pool::run_by_all(Op_func op, void* param) {
    for (int i = 0; i < m_nthreads; i++)
        run_by_one(i, op, param);
}

void OMP_thread_pool::run_by_all_blocked(Op_func op, void* param) {
    for (int i = 0; i < m_nthreads; i++)
        run_by_one(i, op, param);
    wait_for_all();
}

void OMP_thread_pool::run_by_one(int thread_id, Op_func op, void* param) {
    m_job_queues[thread_id].push(Execution_loop_param_pack(Execution_loop_param_pack::WITHOUT_THREAD_INFO, reinterpret_cast<void*>(op), param));
}

Thread_return_value OMP_thread_pool::try_wait_for_one(int thread_id) {
    Thread_return_value ret;
    m_ret_queues[thread_id].try_pop(ret);
    return ret;
}

Thread_return_value OMP_thread_pool::try_wait_for_any() {
    Thread_return_value ret;

    for (int i = 0; i < m_nthreads; i++) {
        ret = wait_for_one(i);
    }

    return ret;
}

Thread_return_value OMP_thread_pool::wait_for_one(int thread_id) {
    Thread_return_value ret;
    m_ret_queues[thread_id].pop(ret);
    return ret;
}


Thread_return_value OMP_thread_pool::wait_for_any() {
    Thread_return_value ret;
    while (ret.get_thread_id() == -1) {
        ret = try_wait_for_any();
    }
    return ret;
}

vector<Thread_return_value> OMP_thread_pool::wait_for_all() {
    vector<Thread_return_value> ret;
    ret.resize(m_nthreads);

    for (int i = 0; i < m_nthreads; i++) {
        ret[i] = wait_for_one(i);
    }

    return ret;
}
