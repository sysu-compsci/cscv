#pragma once

#include <map>

#include "arch/thread_instance_map.hpp"

template <class Time_type = uint64_t>
struct Three_layer_timer {
    std::map<int, std::map<int, std::map<int, Time_type> > > m_timers;  // direction, framework, process

    Time_type& get_time_ref(int key_l0, int key_l1, int key_l2) {
        return m_timers[key_l0][key_l1][key_l2];
    }
};

using Threads_timer_map = Thread_instance_map<Three_layer_timer<>>;
using Timers_map = std::map<int, Three_layer_timer<>*>;
using Timers_instance_map = std::map<int, Three_layer_timer<>>;

static inline Timers_instance_map dump_timers_map_and_clear() {
    Timers_map timers_map = Threads_timer_map::get_instance_map_copy();

    Timers_instance_map ret;

    for (auto& p_tid : timers_map) {
        ret[p_tid.first] = *p_tid.second;
        // p.second->m_timers.clear();

        for (auto& p_direction : p_tid.second->m_timers) {
            for (auto& p_framework : p_direction.second) {
                for (auto& p_process : p_framework.second) {
                    p_process.second = 0;
                }
            }
        }
    }

    return ret;
}

static uint64_t rdtsc_cross() {
#if defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    // 使用内联汇编读取时间戳
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__ARM_ARCH)
    uint64_t virtual_timer_value;
    // 在 ARM 架构中读取虚拟计数器
    asm volatile("mrs %0, cntvct_el0" : "=r" (virtual_timer_value));
    return virtual_timer_value;
#else
#error "Platform not supported!"
#endif
}

static inline uint64_t rdtsc_interval() {
    static __thread uint64_t last_cc;
    uint64_t tmp = last_cc;
    last_cc = rdtsc_cross();
    return last_cc - tmp;
}
