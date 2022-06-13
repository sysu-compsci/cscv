#pragma once

#include <assert.h>
#include <memory.h>
#include <mkl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <sstream>

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif  // defined(__GNUC__)

#include "candy.h"

std::string strprintf(const char *fmt, ...);

template <size_t NUM_DIM>
struct Point_nd {
    int m_coords[NUM_DIM];
    int& operator[](size_t offset) { return m_coords[offset]; }
    // 
    const int& operator[](size_t offset) const { return m_coords[offset]; } 
    std::string to_string() const {
        std::stringstream ss;
        ss << "<";
        for (int i = 0; i < NUM_DIM; i++)
            ss << m_coords[i] << ",";
        ss << ">";
        return ss.str();
    }
};

struct Range_1d {
    int m_start;
    int m_size;

    Range_1d() {}
    Range_1d(int start, int size) : m_start(start), m_size(size) {}

    int get_end() const { return m_start + m_size; }
    int get_last() const { return m_start + m_size - 1; }

    bool contains(int coord) const { return coord >= m_start && coord < (m_start + m_size); }
    bool is_valid() const { return m_size > 0; }
    Range_1d get_intersection(const Range_1d& operand) const {
        Range_1d ret;
        ret.m_start = std::max(m_start, operand.m_start);
        ret.m_size = std::min(m_start + m_size, operand.m_start + operand.m_size) - ret.m_start;

        return ret;
    }
    Range_1d operator&(const Range_1d& operand) const { return get_intersection(operand); }
    std::string to_string() const { return strprintf("<%d %d>", m_start, m_size); }

    void change_start(int new_start) {
        m_size += m_start - new_start;
        m_start = new_start;
    }

    void change_end(int new_end) { m_size += new_end - get_end(); }

    void resize_to_contain_a_point(int offset) {
        if (offset < m_start)
            change_start(offset);
        if (offset > get_end())
            change_end(offset);
    }
};

static inline std::ostream& operator<< (std::ostream& out, Range_1d const & range) {
    out << range.to_string();
    return out;
}

// TODO: enable initializer list for this
// TODO: stringstream
template <size_t NUM_DIM>
struct Range_nd {
    Range_1d m_ranges[NUM_DIM];

    Range_1d& operator[](size_t offset) { return m_ranges[offset]; }
    const Range_1d& operator[](size_t offset) const { return m_ranges[offset]; }
    Range_nd<NUM_DIM> get_intersection(const Range_nd<NUM_DIM>& operand) const;
    Range_nd<NUM_DIM> operator&(const Range_nd<NUM_DIM>& operand) const { return get_intersection(operand); }
    bool is_valid() const;
    bool contains(const Point_nd<NUM_DIM>& point) const;
    std::string to_string() const {
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < NUM_DIM; i++) {
            ss << m_ranges[i].to_string();
        }
        ss << "]";
        return ss.str();
    }

    void resize_to_contain_a_point(const Point_nd<NUM_DIM>& point) {
        for (int i = 0; i < NUM_DIM; i++)
            m_ranges[i].resize_to_contain_a_point(point[i]);
    }
};

template <size_t NUM_DIM>
Range_nd<NUM_DIM> Range_nd<NUM_DIM>::get_intersection(const Range_nd<NUM_DIM>& operand) const {
    Range_nd<NUM_DIM> ret;
    for (int i = 0; i < NUM_DIM; i++)
        ret[i] = m_ranges[i] & operand[i];

    return ret;
}

template <size_t NUM_DIM>
bool Range_nd<NUM_DIM>::is_valid() const {
    bool ret = true;

    for (int i = 0; i < NUM_DIM; i++)
        ret = (ret && m_ranges[i].is_valid());

    return ret;
}

template <size_t NUM_DIM>
bool Range_nd<NUM_DIM>::contains(const Point_nd<NUM_DIM>& point) const {
    bool ret = true;

    for (int i = 0; i < NUM_DIM; i++)
        ret = (ret && m_ranges[i].contains(point[i]));

    return ret;
}

static inline int BLOCK_SIZE(int block_id, int total_blocks, int n) {
    return (n / total_blocks) + ((n % total_blocks > block_id) ? 1 : 0);
}

static inline int BLOCK_LOW(int block_id, int total_blocks, int n) {
    return (n / total_blocks) * block_id + ((n % total_blocks > block_id) ? block_id : n % total_blocks);
}

template <class Out_type, class In_type>
static inline Out_type cast_pointer(In_type in) {
    void* tmp = reinterpret_cast<void*>(&in);
    Out_type* tmp2 = reinterpret_cast<Out_type*>(tmp);
    return *tmp2;
}

struct True_type
{
    static bool value() {return true;}
};

struct False_type
{
    static bool value() {return false;}
};

template <class T1, class T2>
struct Is_same : False_type {};

template <class T>
struct Is_same<T, T> : True_type {};



#ifndef ASSERT_AND_PRINTF

#define ASSERT_AND_PRINTF(expression, format, ...) if (!(expression)) {\
        fprintf(stderr, format, ## __VA_ARGS__);\
        assert(false);\
    }
#endif

#ifndef PRINTF
#define PRINTF(format, ...) fprintf(stdout, "/* at Function %s(), Line %d of %s*/ " format, \
                                    __FUNCTION__, __LINE__, __FILE__, ## __VA_ARGS__);
#endif

#ifndef PRINTF_ERR
#define PRINTF_ERR(format, ...) fprintf(stderr, "/* at Function %s(), Line %d of %s*/ " format, \
                                    __FUNCTION__, __LINE__, __FILE__, ## __VA_ARGS__);
#endif

// #define MALLOC_WITH_CHECK(size) void* ret = malloc(size); if (ret == NULL) PRINTF_ERR("malloc error!\n"); ret;

// cannot implement this as a macro
static inline void* malloc_with_check(unsigned long size) {
    void* ret = malloc(size);
    ASSERT_AND_PRINTF(ret != NULL, "failed to allocate memory with size %lu\n", size);

    return ret;
}

static constexpr int MM_DEFAULT_ALIGNMENT = 512;

static inline void* malloc_with_check(unsigned long size, size_t alignment) {
    void* ret = _mm_malloc(size, alignment);
    ASSERT_AND_PRINTF(ret != NULL, "failed to allocate memory with size %lu\n", size);

    return ret;
}

#ifdef __cplusplus

template <class T>
T* malloc_with_check(unsigned long arr_len) {
    void* ret = malloc(arr_len * sizeof(T));
    ASSERT_AND_PRINTF(ret != NULL, "failed to allocate memory with size %lu\n", arr_len * sizeof(T));

    return reinterpret_cast<T*>(ret);
}

template <class T>
T* malloc_with_check(unsigned long arr_len, size_t alignment) {
    void* ret = _mm_malloc(arr_len * sizeof(T), alignment);
    ASSERT_AND_PRINTF(ret != NULL, "failed to allocate memory with size %lu\n", arr_len * sizeof(T));

    return reinterpret_cast<T*>(ret);
}

#endif

template <class Index_type, class Value_type>
void quick_sort_index_by_value(Index_type *m_nz_indexes, Value_type *m_vals, Index_type l, Index_type r) {
    Index_type i = l, j = r;
    Index_type mid_index = m_nz_indexes[(i + j) / 2], swap_index;
    Value_type swap_val;

    while (i <= j) {
        while (m_nz_indexes[i] < mid_index) i++;
        while (m_nz_indexes[j] > mid_index) j--;
        if (i <= j) {
            swap_val   = m_vals[i];
            swap_index = m_nz_indexes[i];

            m_vals[i] = m_vals[j];
            m_nz_indexes[i] = m_nz_indexes[j];

            m_vals[j] = swap_val;
            m_nz_indexes[j] = swap_index;

            i++; j--;
        }
    }

    if (i < r) quick_sort_index_by_value(m_nz_indexes, m_vals, i, r);
    if (l < j) quick_sort_index_by_value(m_nz_indexes, m_vals, l, j);
}

template <class T>
std::string vector_to_string(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < vec.size(); i++)
        ss << vec[i] << ", ";
    ss << "]";
    return ss.str();
}

// TODO: arbitrary count
template <class T1, class T2>
std::string arrs_to_string(int size, const T1* arr1, const T2* arr2) {
    std::stringstream ss;

    ss << "[";
    for (int i = 0; i < size; i++)
        ss << "<" << arr1[i] << ", " << arr2[i] << ">";
    ss << "]";

    return ss.str();
}

template <class T1>
std::string arr_to_string(int size, const T1* arr) {
    std::stringstream ss;

    ss << "[";
    for (int i = 0; i < size; i++)
        ss << arr[i] << ", ";
    ss << "]";

    return ss.str();
}

template <class T>
T get_arr_min(const T* arr, int size) {
    T ret = arr[0];
    for (int i = 1; i < size; i++)
        ret = std::min(ret, arr[i]);
    return ret;
}

template <class T>
T get_arr_max(const T* arr, int size) {
    T ret = arr[0];
    for (int i = 1; i < size; i++)
        ret = std::max(ret, arr[i]);
    return ret;
}

static inline void __attribute__ ((__gnu_inline__, __always_inline__, __artificial__)) check_mkl_sparse_ret(sparse_status_t ret)  {
    if (ret == SPARSE_STATUS_SUCCESS)
        return;
    if (ret == SPARSE_STATUS_NOT_INITIALIZED) {
        printf("SPARSE_STATUS_NOT_INITIALIZED\n");
        assert(false);
    }
    if (ret == SPARSE_STATUS_ALLOC_FAILED) {
        printf("SPARSE_STATUS_ALLOC_FAILED\n");
        assert(false);
    }
    if (ret == SPARSE_STATUS_INVALID_VALUE) {
        printf("SPARSE_STATUS_INVALID_VALUE\n");
        assert(false);
    }
    if (ret == SPARSE_STATUS_EXECUTION_FAILED) {
        printf("SPARSE_STATUS_EXECUTION_FAILED\n");
        assert(false);
    }
    if (ret == SPARSE_STATUS_INTERNAL_ERROR) {
        printf("SPARSE_STATUS_INTERNAL_ERROR\n");
        assert(false);
    }
    if (ret == SPARSE_STATUS_NOT_SUPPORTED) {
        printf("SPARSE_STATUS_NOT_SUPPORTED\n");
        assert(false);
    }
}

static inline std::string extended_string_by_char(std::string str, int final_size, char c) {
    int ori_size = str.size();
    if (final_size <= ori_size)
        return str;
    str.resize(final_size);
    for (int i = ori_size; i < final_size; i++) {
        str[i] = c;
    }

    return str;
}

static inline void align_strings_by_char(std::string* strings, int count, char c) {
    int max_length = 0;
    for (int i = 0; i < count; i++)
        if (strings[i].size() > max_length)
            max_length = strings[i].size();

    for (int i = 0; i < count; i++) {
        strings[i] = extended_string_by_char(strings[i], max_length, c);
    }
}
