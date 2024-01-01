#pragma once

#include <tbb/concurrent_queue.h>
#include <atomic>

class Ring_fake_barrier {
private:
    std::atomic<uint64_t> **m_flags;
    int m_nthreads;

public:
    Ring_fake_barrier(int nthreads);
    ~Ring_fake_barrier();

    // run pass before wait!
    void pass(int tid);
    void wait(int tid);

    int get_nthreads() const { return m_nthreads; }
};
