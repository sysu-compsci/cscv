#pragma once

#include <map>

#include "arch/pthread_util.hpp"
#include "base/basic_definition.hpp"

template <class Instance_type>
class Thread_instance_map {
    Thread_instance_map(const Thread_instance_map&);

    pthread_spinlock_t m_lock;

    Thread_instance_map() {
        pthread_spin_init(&m_lock, 0);
    }
    ~Thread_instance_map() {
        pthread_spin_destroy(&m_lock);
    }

    static std::map<int, Instance_type*>& get_registered_instance_map() {
        static std::map<int, Instance_type*> instance_map;
        return instance_map;
    }

    static Thread_instance_map& get_instance() {
        static Thread_instance_map instance;
        return instance;
    }
    
    static Instance_type*& get_thread_ptr_ref() {
        static __thread Instance_type* thread_local_ptr = nullptr;
        return thread_local_ptr;
    }

public:
    // how to avoid this?
    static void init_in_sequential_area() {
        get_instance();
    }

    static void register_thread(int thread_id) {
        Spinlock_guard lock_guard(&get_instance().m_lock);

        if (get_registered_instance_map().count(thread_id) != 0) {
            delete get_registered_instance_map()[thread_id];
        }

        get_thread_ptr_ref() = new Instance_type;
        get_registered_instance_map()[thread_id] = get_thread_ptr_ref();
    }

    // called by main thread
    static std::map<int, Instance_type*> get_instance_map_copy() {
        return get_registered_instance_map();
    }

    void clear() {
        for (auto p : get_registered_instance_map()) {
            delete p.second;
        }
        get_registered_instance_map().clear();
    }

    static Instance_type* get_local() {
        ASSERT_AND_PRINTF(get_thread_ptr_ref() != nullptr, "Unregistered thread!\n");
        return get_thread_ptr_ref();
    }
};
