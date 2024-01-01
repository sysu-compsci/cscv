#include "ring_fake_barrier.hpp"

Ring_fake_barrier::Ring_fake_barrier(int nthreads) : m_nthreads(nthreads) {
    m_flags = new std::atomic<uint64_t> *[nthreads];

    for (int tid = 0; tid < m_nthreads; tid++) {
        m_flags[tid] = new std::atomic<uint64_t>;
        m_flags[tid][0] = 0;
    }
}

Ring_fake_barrier::~Ring_fake_barrier() {
    for (int tid = 0; tid < m_nthreads; tid++) {
        delete m_flags[tid];
    }
    delete m_flags;
}

void Ring_fake_barrier::pass(int tid) {
    m_flags[tid][0]++;
}

void Ring_fake_barrier::wait(int tid) {
    int next_tid = (tid + 1) % m_nthreads;
    while (m_flags[tid][0] > m_flags[next_tid][0]);
}
