#pragma once

#include <immintrin.h>
#include <mkl.h>

#include "base/basic_definition.hpp"
#include "data/data_container.hpp"
#include "cscv/flags.hpp"

// #define NO_KERNEL  // to save compilation time

#if !defined(__AVX512F__)
constexpr bool c_vexpand_enabled = false;
// for compilation

static inline __m512 vexpand_loadu_floatv16 (__mmask16 __U, void const *__P) { return __m512(); }
static inline __m256 vexpand_loadu_floatv8 (__mmask8 __U, void const *__P) { return __m256(); }
static inline __m128 vexpand_loadu_floatv4 (__mmask8 __U, void const *__P) { return __m128(); }
static inline __m512d vexpand_loadu_doublev8 (__mmask8 __U, void const *__P) { return __m512d(); }
static inline __m256d vexpand_loadu_doublev4 (__mmask8 __U, void const *__P) { return __m256d(); }

static inline void vstore_floatv16 (void *__P, __m512 __A) {}
static inline void vstore_floatv8 (float *__P, __m256 __A) {}
static inline void vstore_floatv4 (float *__P, __m128 __A) {}
static inline void vstore_doublev8 (void *__P, __m512d __A) {}
static inline void vstore_doublev4 (double *__P, __m256d __A) {}

#else

constexpr bool c_vexpand_enabled = true;
#define vexpand_loadu_floatv16 _mm512_maskz_expandloadu_ps
#define vexpand_loadu_floatv8 _mm256_maskz_expandloadu_ps
#define vexpand_loadu_floatv4 _mm_maskz_expandloadu_ps
#define vexpand_loadu_double8 _mm512_maskz_expandloadu_pd
#define vexpand_loadu_double4 _mm256_maskz_expandloadu_pd

#define vstore_floatv16 _mm512_store_ps
#define vstore_floatv8 _mm256_store_ps
#define vstore_floatv4 _mm_store_ps
#define vstore_doublev8 _mm512_store_pd
#define vstore_doublev4 _mm256_store_pd

#endif

template <class Element_type, uint8_t t_vec_size>
struct Vec_expand_mask;

template <>
struct Vec_expand_mask<double, 2> {
    using type = uint8_t;
};

template <>
struct Vec_expand_mask<double, 4> {
    using type = uint8_t;
};

template <>
struct Vec_expand_mask<double, 8> {
    using type = uint8_t;
};

// this should not happen!
template <>
struct Vec_expand_mask<double, 16> {
    using type = uint16_t;
};

template <>
struct Vec_expand_mask<float, 4> {
    using type = uint8_t;
};

template <>
struct Vec_expand_mask<float, 8> {
    using type = uint8_t;
};

template <>
struct Vec_expand_mask<float, 16> {
    using type = uint16_t;
};

template <class Element_type, uint8_t t_vec_size, uint16_t t_byte_alignment = get_2_power_ceil(t_vec_size * sizeof(Element_type))>
struct FP_vec {
    static_assert(std::is_same<Element_type, float>::value || std::is_same<Element_type, double>::value, "only float and double are supported");

    // TODO: disable the use of double v16 at runtime, since not it is not natively supported

    using Mask_type = typename Vec_expand_mask<Element_type, t_vec_size>::type;

    FP_vec() {}

    FP_vec(Element_type val) {
        set1(val);
    }

    FP_vec(const Element_type* source) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        for (int i = 0; i < t_vec_size; i++)
            m_scalar[i] = source[i];
    }

    inline FP_vec<Element_type, t_vec_size, t_byte_alignment> operator *(const FP_vec<Element_type, t_vec_size, t_byte_alignment>& right) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        __builtin_assume_aligned(&right.m_scalar, t_byte_alignment);
        FP_vec<Element_type, t_vec_size, t_byte_alignment> ret;
        for (int i = 0; i < t_vec_size; i++) {
            ret.m_scalar[i] = m_scalar[i] * right.m_scalar[i];
        }
        return ret;
    }

    inline void operator += (const FP_vec<Element_type, t_vec_size, t_byte_alignment>& right) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        __builtin_assume_aligned(&right.m_scalar, t_byte_alignment);
        for (int i = 0; i < t_vec_size; i++) {
            m_scalar[i] += right.m_scalar[i];
        }
    }

    inline void mad_self(const FP_vec<Element_type, t_vec_size, t_byte_alignment>& a, const FP_vec<Element_type, t_vec_size, t_byte_alignment>& b) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        __builtin_assume_aligned(&a.m_scalar, t_byte_alignment);
        __builtin_assume_aligned(&b.m_scalar, t_byte_alignment);
        for (int i = 0; i < t_vec_size; i++) {
            m_scalar[i] += a.m_scalar[i] * b.m_scalar[i];
        }
    }

    Element_type& at(int index) {
        // ASSERT_AND_PRINTF(index < t_vec_size & index >= 0, "%d out of vec range %d\n", index, t_vec_size);
        return m_scalar[index];
    }

    const Element_type& at(int index) const {
        // ASSERT_AND_PRINTF(index < t_vec_size & index >= 0, "%d out of vec range %d\n", index, t_vec_size);
        return m_scalar[index];
    }

    Element_type& operator[](size_t offset) { return at(offset); }
    const Element_type& operator[](size_t offset) const { return at(offset); }

    inline void set1(Element_type val) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        for (int i = 0; i < t_vec_size; i++)
            m_scalar[i] = val;
    }

    inline Element_type get_sum() const {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        Element_type ret = 0;
        for (int i = 0; i < t_vec_size; i++)
            ret += m_scalar[i];
        return ret;
    }

    inline void expand_from_memory(Mask_type mask, Element_type const* mem_addr) {
        __builtin_assume_aligned(m_scalar, t_byte_alignment);
        Mask_type shift_mask = 1;
        Mask_type src_offset = 0;

        const FP_vec<Element_type, t_vec_size, t_byte_alignment>* a = (const FP_vec<Element_type, t_vec_size, t_byte_alignment>*)(mem_addr);

        for (int i = 0; i < t_vec_size; i++) {
            if (shift_mask & mask) {
                at(i) = a->at(src_offset);
                src_offset++;
            } else {
                at(i) = 0;
            }
            shift_mask <<= 1;
        }
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "[";

        for (int i = 0; i < t_vec_size; i++) {
            ss << at(i) << ",";
        }

        ss << "]";

        return ss.str();
    }

// private:
    Element_type m_scalar[t_vec_size] __attribute__((aligned(t_byte_alignment)));
} __attribute__((aligned(t_byte_alignment)));
