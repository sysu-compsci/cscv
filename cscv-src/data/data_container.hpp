#pragma once
#if defined(__x86_64__) || defined(__i386__)
#include <mkl.h>
#endif

#include <string>

#include "base/basic_definition.hpp"

// typedef struct CSC_matrix_struct
// {
// 	int* m_col_offsets;
// 	int* m_row_idxs;
// 	float* m_vals;

// 	int m_num_row, m_num_col;
// 	int m_sub_start_row, m_sub_start_col;
// 	int m_nz_count;
// } CSC_matrix_c;

template <class Element_type>
class Dense_vector {
    template <class Src_element_type>
    friend class Dense_vector;

private:
    Element_type* m_data;
    bool m_self_allocated = false;
    size_t m_size;

public:
    explicit Dense_vector(size_t size, Element_type* data) : m_size(size), m_data(data), m_self_allocated(false) {}

    // must add explict here!
    explicit Dense_vector(size_t size) : m_size(size), m_self_allocated(true) {
        m_data = malloc_with_check<Element_type>(m_size, MM_DEFAULT_ALIGNMENT);
    }

    template <class Src_element_type>
    explicit Dense_vector(Dense_vector<Src_element_type>* src) {
        // force type casting
        m_data = (Element_type*)src->m_data;
        m_self_allocated = false;
        m_size = (sizeof(Src_element_type) * src->m_size) / sizeof(Element_type);
        ASSERT_AND_PRINTF(get_bytes() == src->get_bytes(), "bytes mismatch!\n");
    }

    Element_type& operator[](size_t offset) {
        ASSERT_AND_PRINTF(offset < m_size, "offset = %lu, while vec size = %lu\n", offset, m_size);
        return m_data[offset];
    }

    // how to handle negative index? (extreme condition when padded y lhs is less than 9)
    Element_type& at(size_t offset) const {
        ASSERT_AND_PRINTF(offset < m_size, "offset = %lu, while vec size = %lu\n", offset, m_size);
        return m_data[offset];
    }
    
    // const Element_type& at(size_t offset) const {
    //     ASSERT_AND_PRINTF(offset < m_size, "offset = %d, while vec size = %d\n", offset, m_size);
    //     return m_data[offset];
    // }

    ~Dense_vector() {
        if (m_self_allocated)
            _mm_free(m_data);
    }

    void set_zero() const {
        memset(m_data, 0, sizeof(Element_type) * m_size);
    }

    void set_zero_omp_for() {
        #pragma omp for nowait
        for (size_t i = 0; i < m_size; i++)
            m_data[i] = 0;
    }

    size_t get_size() const { return m_size; }
    size_t get_bytes() const { return m_size * sizeof(Element_type); }
};

/**
 * COO_diff:
 *      Description:
 *          COO_diff is implemented to represent the mismatch of elements between two matrixes.
 **/
struct COO_diff {
    enum Diff_type  {
        UNEQUAL = 0,  // the value of elements in two matrixes is not equal
        LEFT_EMPTY = 1,  // the element only exists in the left matrix
        RIGHT_EMPTY = 2  // the element only exists in the right matrix
    };
    static constexpr double EMPTY_VALUE = -999999;

    int m_diff_type, m_row, m_col;
    double m_right_val, m_left_val;
    COO_diff() {}
    COO_diff(int diff_type, int row, int col, double right_val, double left_val) :
    m_diff_type(diff_type), m_row(row), m_col(col), m_right_val(right_val), m_left_val(left_val) {}
};

template <class Element_type>
struct CSR_matrix {
private:
    CSR_matrix(const CSR_matrix&);

public:
    CSR_matrix(int num_row, int num_col, int sub_start_row, int sub_start_col, int nz_count);
    ~CSR_matrix();

    void diff_to_arr(const CSR_matrix<Element_type>& right, std::vector<COO_diff>& diff);
    void diff_to_file(const CSR_matrix<Element_type>& right);

    int get_num_row() const {return m_num_row;}
    int get_num_col() const {return m_num_col;}
    int get_sub_start_row() const {return m_sub_start_row;}
    int get_sub_start_col() const {return m_sub_start_col;}
    uint64_t get_bytes() const {
        return sizeof(int) * (m_num_row + 1 + m_nz_count) + sizeof(Element_type) * m_nz_count;
    }

    void multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);
    void multiply_dense_vector_trans(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);

    #if defined(__x86_64__) || defined(__i386__)
    sparse_matrix_t* convert_to_mkl_matrix();
    sparse_matrix_t* convert_to_mkl_matrix_csc_trans();
    #endif

// private:
    int m_num_row, m_num_col;
    int m_sub_start_row, m_sub_start_col;
    int m_nz_count;

    int *m_col_idxs, *m_row_offsets;
    Element_type *m_vals;
};


template <class Element_type>
struct CSC_matrix {
private:
    CSC_matrix(const CSC_matrix&);

public:
    CSC_matrix(int num_row, int num_col, int sub_start_row, int sub_start_col, int nz_count);
    ~CSC_matrix();

    int get_num_row() const {return m_num_row;}
    int get_num_col() const {return m_num_col;}
    int get_sub_start_row() const {return m_sub_start_row;}
    int get_sub_start_col() const {return m_sub_start_col;}
    uint64_t get_bytes() const {
        return sizeof(int) * (m_num_col + 1 + m_nz_count) + sizeof(Element_type) * m_nz_count;
    }

    void multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);
    void multiply_dense_vector_trans(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);

#if defined(__x86_64__) || defined(__i386__)
    sparse_matrix_t* convert_to_mkl_matrix();
    sparse_matrix_t* convert_to_mkl_matrix_csr_trans();
#endif

    bool m_flag_row_sorted = false;

    int m_num_row, m_num_col;
    int m_sub_start_row, m_sub_start_col;
    int m_nz_count;

    int *m_row_idxs, *m_col_offsets;
    Element_type *m_vals;

    int get_col_nnz(int local_col);
};

/**
 * COO_matrix_buffer:
 *      Description:
 *          COO_matrix_buffer act as a list of tuple<row, col, value> for a COO matrix.
 *          Also, it contains the sub-matrix information of size and offset.
 **/
template <class Element_type>
struct COO_matrix_buffer {
    int m_num_row, m_num_col;
    int m_sub_start_row, m_sub_start_col;
    int m_nz_count;

    std::vector<int>        m_nz_row_idx;
    std::vector<int>        m_nz_col_idx;
    std::vector<Element_type> m_vals;

    std::vector<int>         m_row_nz_count, m_col_nz_count;

    COO_matrix_buffer(int num_row, int num_col, int sub_start_row, int sub_start_col);
    ~COO_matrix_buffer();

    void recount_nz_count();

    void add_element(int global_row, int global_col, Element_type val);
    template <class Sub_element_type = Element_type>
    void add_sub_element(int sub_row, int sub_col, Sub_element_type val);

    std::string get_summary_string() const {
        return strprintf("m_num_row = %d m_num_col = %d m_sub_start_row = %d m_sub_start_col = %d m_nz_count = %d,"
                         "checksum = %.12g",
                          m_num_row, m_num_col, m_sub_start_row, m_sub_start_col, m_nz_count, get_checksum());
    }

    void add_coo_buffer(const COO_matrix_buffer&);

    bool check_in_range(int global_row_idx, int global_col_idx);

    template <class Out_element_type = Element_type>
    CSR_matrix<Out_element_type>* convert_to_csr_matrix() const;
    template <class Out_element_type = Element_type>
    CSC_matrix<Out_element_type>* convert_to_csc_matrix() const;

    double get_checksum() const;
    void resolve(CSR_matrix<Element_type>* csr);
    void resolve(COO_matrix_buffer<Element_type>* coo);

    void multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);
    void multiply_dense_vector_trans(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec);

    static COO_matrix_buffer<Element_type>* read_from_file(std::string filename);
    bool dump_to_file(std::string filename);

private:
    COO_matrix_buffer(const COO_matrix_buffer&);
};

#include "data_container.tcc"
