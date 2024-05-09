#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <string.h>
#include <iostream>
#include <assert.h>
// #include <numa.h>

namespace std {
struct CPUInfoUtil {
    int m_global_core_count, m_global_ht_count;
    std::vector<int> m_global_core_id_by_global_ht_id;
    std::map<int, int> m_global_core_level_by_global_core_id;
    std::vector<int> m_local_core_id_by_local_ht_id;

    std::map<int, int> m_local_core_id_by_global_core_id;
    std::map<int, int> m_global_core_id_by_local_core_id;

    std::map<int, int> m_local_ht_id_by_global_ht_id;
    std::map<int, int> m_global_ht_id_by_local_ht_id;

    static const CPUInfoUtil& I() {
        return GetInstance();
    }

    static const CPUInfoUtil& GetInstance() {
        static CPUInfoUtil cpuinfo_single_instance;
        return cpuinfo_single_instance;
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

        int num = GetInstance().m_global_ht_count;

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

private:
    CPUInfoUtil();
    CPUInfoUtil(const CPUInfoUtil&);  // not to def
    CPUInfoUtil& operator=(const CPUInfoUtil&);  // not to def
    ~CPUInfoUtil() {}
};


};  // namespace std
