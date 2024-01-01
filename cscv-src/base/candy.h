#pragma once


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>


#include <cstdint>

static inline int div_and_ceil(double denominator, int divisor) {
    return ceil(denominator / divisor) + 0.0001;
}

static inline int get_padded_size(int original_size, int alignment) {
    return div_and_ceil(original_size, alignment) * alignment;
}

template <class T>
static inline size_t constexpr get_2_power_ceil(T val) {
    size_t ret = 1;
    while (val > ret)
        ret *= 2;
    return ret;
}

template <class Out_type, class In_type>
Out_type force_cast_pointer(In_type in) {
    void* tmp = reinterpret_cast<void*>(&in);
    Out_type* tmp2 = reinterpret_cast<Out_type*>(tmp);
    return *tmp2;
}

typedef void*(*Pthread_func)(void*);
