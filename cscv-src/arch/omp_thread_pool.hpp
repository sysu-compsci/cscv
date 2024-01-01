#pragma once

#include <pthread.h>

#include <vector>

#include "arch/pthread_util.hpp"
#include "base/basic_definition.hpp"

/**
 * 每个线程都可以提供返回值。
 * 最简单的返回值。
 **/

class Thread_return_value {
    int m_thread_id;  // 哪个线程给的返回值。如果线程号是 -1，说明未初始化。

public:
    Thread_return_value(int thread_id) : m_thread_id(thread_id) {}
    Thread_return_value() : m_thread_id(-1) {}
    int get_thread_id() const { return m_thread_id; }
    // virtual ~Thread_return_value() {}
    // virtual int get_type() { return -1; }
};

/**
 * 先实现一个最重量级的线程池
 * 如果需要更好的性能，在通过特定的 flag 进行流程精简就好了。
 **/
class OMP_thread_pool {
    struct Execution_loop_param_pack {
        enum Execution_param_type {
            WITHOUT_THREAD_INFO = 1,
            TERMINATE
        };

        Execution_param_type m_type;
        void* m_func_ptr;
        void* m_func_param;

        Execution_loop_param_pack() {}
        Execution_loop_param_pack(Execution_param_type type, void* func_ptr, void* func_param) :
                             m_type(type), m_func_ptr(func_ptr), m_func_param(func_param) {}
    };

    int m_nthreads;
    pthread_t m_main_thread;
    std::vector<Thread_safe_queue<Execution_loop_param_pack> > m_job_queues;
    std::vector<Thread_safe_queue<Thread_return_value> > m_ret_queues;
    bool m_active;

    uint64_t m_stack_size;

    OMP_thread_pool(const OMP_thread_pool&);

    struct Execution_pthread_param {
        OMP_thread_pool* m_this;
        int m_thread_id;
        Execution_pthread_param(OMP_thread_pool* _this, int _thread_id) : m_this(_this), m_thread_id(_thread_id) {}
    };

    void main_execution_loop();
    void execution_loop(int thread_id);
    void join();

public:
    typedef void(*Op_func)(void* param);

    OMP_thread_pool(int nthreads, int stack_size_mb);
    ~OMP_thread_pool();

    void run_by_all(Op_func op, void* param);
    void run_by_all_blocked(Op_func op, void* param);
    void run_by_one(int thread_id, Op_func op, void* param);

    Thread_return_value try_wait_for_one(int thread_id);
    Thread_return_value try_wait_for_any();
    Thread_return_value wait_for_one(int thread_id);
    Thread_return_value wait_for_any();
    std::vector<Thread_return_value> wait_for_all();

    void get_nthreads();
};
