#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <numa.h>

namespace std {
struct CPUInfoUtil {
    int total_core_count_, total_socket_count_, core_per_socket_;
    int total_thread_count_, thread_per_socket_, thread_per_core_;

    std::vector<int> raw_siblings_;  // thread count per socket
    std::vector<int> raw_cpu_cores_;  // core count per socket; need to be the same
    std::vector<int> raw_core_id_;  // core id may not be continual. A 10-core processor may have a core id of 12
    std::vector<int> raw_physical_id_;  // socket id
    std::vector<int> raw_processor_;  // logical core id

    // thread level mapping info
    std::vector<int> socket_id_by_thread_;
    std::vector<int> continual_core_id_in_socket_by_thread_;
    std::vector<int> continual_global_core_id_by_thread_;

    std::map<int, int> core_id_map_;  // eg. {0, 1, 3, 4} -> {0, 1, 2, 3}
    // how much tid associated with a physical id
    std::map<int, int> socket_proc_count_map_;  // <socket_id, logical_core_count>
    // the actual tid associated with a socket
    std::map<int, std::vector<int>> threads_by_socket_;  // <socket, vec<logical cores>>
    // ht_id: from 0 to HT_COUNT_PER_CORE-1
    std::map<int, std::vector<std::vector<int>>> threads_by_socket_ht_;  // [socket][ht_id] -> vec<logical cores>

    // the core, to what tid. rely on core_in_global_mapping_
    std::vector<std::vector<int>> threads_by_continual_global_core_id_;

public:
    static const CPUInfoUtil& I() {
        return GetInstance();
    }

    static const CPUInfoUtil& GetInstance() {
        static CPUInfoUtil cpuinfo_single_instance;
        return cpuinfo_single_instance;
    }

    static void SetMembindNode(int node) {
        if(node < 0)
            return;
        struct bitmask *bmp;

        bmp = numa_allocate_nodemask();

        numa_bitmask_setbit(bmp, node);
        numa_set_membind(bmp);

        numa_bitmask_free(bmp);
    }

    static void SetPreferredNode(int node) {
        numa_set_preferred(node);
    }

    static void BindToCore(size_t logical_core) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(logical_core, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
            std::cout << "Failed to set affinity (logical_core: " << logical_core << ")" << endl;
    }

    static std::vector<int> TestBoundLogicalCore() {
        cpu_set_t mask;
        cpu_set_t get;
        int i, j, k;

        vector<int> to_ret;

        int num = GetInstance().total_thread_count_;

        CPU_ZERO_S(sizeof(cpu_set_t), &get);
        if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0) {
            fprintf(stderr, "get thread affinity failed\n");
        }

        for (j = 0; j < num; j++) {
            if (CPU_ISSET_S(j, sizeof(cpu_set_t), &get)) {
                to_ret.push_back(j);
            }
        }

        return to_ret;
    }

    static int get_bound_numa_node() {
        auto bound_list = TestBoundLogicalCore();

        if (bound_list.size() == 0) {
            printf("affinity bound list cannot be empty!\n");
            assert(false);
        }

        int first_node = I().raw_physical_id_[bound_list[0]];

        for (auto node_id : bound_list) {
            if (I().raw_physical_id_[node_id] != first_node) {
                return -1;
            }
        }

        return first_node;
    }

private:
    CPUInfoUtil();
    CPUInfoUtil(const CPUInfoUtil&);  // not to def
    CPUInfoUtil& operator=(const CPUInfoUtil&);  // not to def
    ~CPUInfoUtil() {}
};


};  // namespace std
