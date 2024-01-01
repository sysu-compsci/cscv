#pragma once

#include <pthread.h>
#include <stdio.h>

#include <queue>

using std::queue;

class Spinlock_guard {
public:
    explicit Spinlock_guard(pthread_spinlock_t* lock) {
        m_unlocked = false;

        m_lock = lock;
        if (m_lock != NULL)
            pthread_spin_lock(lock);
    }

    ~Spinlock_guard() {
        unlock();
    }

    void unlock() {
        if (!m_unlocked) {
            m_unlocked = true;
            if (m_lock != NULL)
                pthread_spin_unlock(m_lock);
        }
    }

private:
    pthread_spinlock_t* m_lock;
    bool m_unlocked;
};

class Mutex_guard {
public:
    explicit Mutex_guard(pthread_mutex_t *lock) {
        m_unlocked = false;

        m_lock = lock;
        if (m_lock != NULL) {
            pthread_mutex_lock(m_lock);
        }
    }

    ~Mutex_guard() {
        unlock();
    }

    void unlock() {
        if (!m_unlocked) {
            m_unlocked = true;
            if (m_lock != NULL)
                pthread_mutex_unlock(m_lock);
        }
    }

private:
    pthread_mutex_t* m_lock;
    bool m_unlocked;
};

// 之后再改成定长循环队列
template<class T>
class Thread_safe_queue {
protected:
    queue<T> m_queue;
    // pthread_spinlock_t* m_spinlock;
    pthread_mutex_t m_mutex;
    pthread_cond_t m_cond;


public:
    Thread_safe_queue() {
        // m_spinlock = new pthread_spinlock_t;
        // pthread_spin_init(m_spinlock, 0);
        pthread_mutex_init(&m_mutex, NULL);
        pthread_cond_init(&m_cond, NULL);
    }

    // just implemented for the fast initialization in stl containers
    Thread_safe_queue(const Thread_safe_queue&) {
        // m_spinlock = new pthread_spinlock_t;
        // pthread_spin_init(m_spinlock, 0);
        pthread_mutex_init(&m_mutex, NULL);
        pthread_cond_init(&m_cond, NULL);
    }

    ~Thread_safe_queue() {
        // pthread_spin_destroy(m_spinlock);
        // delete m_spinlock;
        pthread_mutex_destroy(&m_mutex);
        pthread_cond_destroy(&m_cond);
    }

    bool try_pop(T& data) {
        // Spinlock_guard lock_guard(m_spinlock);
        Mutex_guard lock_guard(&m_mutex);
        if (m_queue.size() == 0)
            return false;

        data = m_queue.front();
        m_queue.pop();
        return true;
    }

    bool push(const T& data) {
        // Spinlock_guard lock_guard(m_spinlock);
        {
            Mutex_guard lock_guard(&m_mutex);
            m_queue.push(data);
        }
        pthread_cond_broadcast(&m_cond);

        // 如果是循环队列，并且 overflow 了，则会 return false
        return true;
    }

    void pop(T& data) {
        Mutex_guard lock_guard(&m_mutex);
        while (true) {
            if (m_queue.size() == 0) {
                pthread_cond_wait(&m_cond, &m_mutex);
            } else {
                data = m_queue.front();
                m_queue.pop();
                break;
            }
        }
    }
};

class Barrier_pthread_naive {
    int m_target;

    mutable int m_arrived;
    mutable pthread_mutex_t m_mutex;
    mutable pthread_cond_t m_cond;

public:
    Barrier_pthread_naive(int num_thread);
    ~Barrier_pthread_naive();

    void sync() const;
};
