#pragma once

#include <omp.h>

#include "arch/cpuinfo_util.hpp"
#include "arch/pthread_util.hpp"
#include "base/basic_definition.hpp"

class Naive_seq_mem_pool {
    char* m_mem_pool;

    uint64_t m_size, m_offset;

    pthread_spinlock_t m_lock;

public:
    void reset() { m_offset = 0; }
    char* allocate(uint64_t size);

    Naive_seq_mem_pool(uint64_t size);
    ~Naive_seq_mem_pool();
};

class Naive_NUMA_util {
private:
    const std::CPUInfoUtil& m_cpuinfo_util = std::CPUInfoUtil::I();

    uint64_t m_total_mempool_size, m_single_mempool_size;
    int m_node_count;

    std::vector<Naive_seq_mem_pool*> m_mempool_on_nodes;

    void create_mempool_with_locality(int node_id);  // do not call this in main thread

    Naive_NUMA_util();
    Naive_NUMA_util(const Naive_NUMA_util&);
    Naive_NUMA_util& operator=(const Naive_NUMA_util&);  // not to def
    ~Naive_NUMA_util();

public:
    static Naive_NUMA_util& get_instance() {
        static Naive_NUMA_util instance;
        return instance;
    }

    void create_seq_mem_pool(uint64_t size);  // these sizes will be distributed on different numa nodes
    bool seq_mem_pool_exist() const { return m_mempool_on_nodes.size() != 0; }
    char* allocate_with_numa_locality(uint64_t size);  // call on a thread with specific

    void print_bound_info_in_omp_parallel_region();
};
