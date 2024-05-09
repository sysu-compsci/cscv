#include "naive_numa_util.hpp"

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include <thread>


using namespace std;

char* Naive_seq_mem_pool::allocate(uint64_t size) {
    // Spinlock_guard guard(&m_lock);

    // char* ret = m_mem_pool + m_offset;
    // m_offset += size;

    // ASSERT_AND_PRINTF(m_offset <= m_size, "mempool exsausted! offset = %lu, size = %lu\n", m_offset, m_size);

    // return ret;
}

Naive_seq_mem_pool::Naive_seq_mem_pool(uint64_t size) {
    // m_size = size;
    // m_offset = 0;
    // m_mem_pool = (char*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    // memset(m_mem_pool, 0, size);

    // pthread_spin_init(&m_lock, 0);
}

Naive_seq_mem_pool::~Naive_seq_mem_pool() {
    // pthread_spin_destroy(&m_lock);
    // munmap(m_mem_pool, m_size);
}

Naive_NUMA_util::Naive_NUMA_util() {
    // m_numa_detector = &Naive_NUMA_detector::get_instance();
    // m_node_count = m_numa_detector->get_node_count();

    // m_node_count = CPUInfoUtil::I().total_socket_count_;
}

Naive_NUMA_util::~Naive_NUMA_util() {
    // for (auto* mempool : m_mempool_on_nodes)
    //     delete mempool;
}

void Naive_NUMA_util::create_mempool_with_locality(int node_id) {
    // auto local_logical_cores = CPUInfoUtil::I().threads_by_socket_.at(node_id);

    // CPUInfoUtil::BindToCore(local_logical_cores[0]);

    // m_mempool_on_nodes[node_id] = new Naive_seq_mem_pool(m_single_mempool_size);
}

void Naive_NUMA_util::create_seq_mem_pool(uint64_t size) {
    // // create threads with affinity, and touch them

    // m_total_mempool_size = size;
    // m_single_mempool_size = size / m_node_count;

    // printf("[Naive_NUMA_util::create_seq_mem_pool] Created %d mem pools, total size = %lu, single size = %lu\n",
    //        m_node_count, m_total_mempool_size, m_single_mempool_size);

    // m_mempool_on_nodes.resize(m_node_count);

    // vector<thread> threads;

    // for (int i = 0; i < m_node_count; i++) {
    //     threads.emplace_back(&Naive_NUMA_util::create_mempool_with_locality, this, i);
    // }

    // for (auto& thread : threads) {
    //     thread.join();
    // }
}

char* Naive_NUMA_util::allocate_with_numa_locality(uint64_t size) {
    // int numa_node = CPUInfoUtil::get_bound_numa_node();

    // ASSERT_AND_PRINTF(numa_node >= 0, "allocation with locality failed!\n");
    // ASSERT_AND_PRINTF(numa_node < m_mempool_on_nodes.size(), "mem pool [0, %lu] for numa node %d not allocated\n",
    //                                                          m_mempool_on_nodes.size(), numa_node);

    // return m_mempool_on_nodes[numa_node]->allocate(size);
}

void Naive_NUMA_util::print_bound_info_in_omp_parallel_region() {
    // int nthreads = omp_get_num_threads();
    // int tid = omp_get_thread_num();
    // auto bound_list = CPUInfoUtil::TestBoundLogicalCore();
    // int bound_node = CPUInfoUtil::get_bound_numa_node();

    // for (int i = 0; i < nthreads; i++) {
    //     if (tid == i) {
    //         printf("the affinity list of thread %d (of %d) is %s, on socket %d\n",
    //             tid, nthreads, vector_to_string(bound_list).c_str(), bound_node);
    //     }
    //     #pragma omp barrier
    // }
}
