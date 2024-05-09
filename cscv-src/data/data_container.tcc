#include "data_container.hpp"

#include <map>
#include <cmath>
#include <vector>

template <class Element_type>
CSR_matrix<Element_type>::CSR_matrix(int num_row, int num_col, int sub_start_row, int sub_start_col, int nz_count) :
                        m_num_row(num_row), m_num_col(num_col), m_sub_start_row(sub_start_row), m_sub_start_col(sub_start_col), m_nz_count(nz_count) {
    m_row_offsets = malloc_with_check<int>(m_num_row + 1, MM_DEFAULT_ALIGNMENT);  // reinterpret_cast<int*>(malloc(sizeof(int) * (m_num_row + 1)));
    m_col_idxs = malloc_with_check<int>(m_nz_count, MM_DEFAULT_ALIGNMENT);  // reinterpret_cast<int*>(malloc(sizeof(int) * m_nz_count));
    m_vals = malloc_with_check<Element_type>(m_nz_count, MM_DEFAULT_ALIGNMENT); // reinterpret_cast<Out_element_type*>(malloc(sizeof(Out_element_type) * m_nz_count));
}

template <class Element_type>
CSR_matrix<Element_type>::~CSR_matrix() {
    _mm_free(m_col_idxs);
    _mm_free(m_row_offsets);
    _mm_free(m_vals);
}

template <class Element_type>
void CSR_matrix<Element_type>::diff_to_arr(const CSR_matrix<Element_type>& right, std::vector<COO_diff>& diff) {
    // dump the diff to a file
    if (m_num_row != right.m_num_row || m_num_col != right.m_num_col)
        return;

    FILE *outf = fopen("csr_diff.txt", "w");

    for (int row_id = 0; row_id < m_num_row; row_id++) {
        std::map<int, Element_type> my_val_map, right_val_map;
        for (int index = m_row_offsets[row_id]; index < m_row_offsets[row_id + 1]; index++)
            my_val_map[m_col_idxs[index]] = m_vals[index];
        for (int index = right.m_row_offsets[row_id]; index < right.m_row_offsets[row_id + 1]; index++)
            right_val_map[right.m_col_idxs[index]] = right.m_vals[index];

        // the index is sorted in the map
        auto my_it = my_val_map.begin(), right_it = right_val_map.begin();

        // compare each row
        while (true) {
            // compare
            const Element_type eps = 0;

            if (my_it == my_val_map.end() && right_it == right_val_map.end())
                break;

            if (my_it == my_val_map.end()) {
                diff.push_back(COO_diff(COO_diff::LEFT_EMPTY, row_id, right_it->first,
                                        COO_diff::EMPTY_VALUE, right_it->second));
                right_it++;
            } else if (right_it == right_val_map.end()) {
                diff.push_back(COO_diff(COO_diff::RIGHT_EMPTY, row_id, my_it->first, my_it->second,
                                        COO_diff::EMPTY_VALUE));
                my_it++;
            } else if (my_it->first == right_it->first) {
                if (abs(my_it->second - right_it->second) > eps) {
                    // record diff
                    diff.push_back(COO_diff(COO_diff::UNEQUAL, row_id, my_it->first,
                                            my_it->second, right_it->second));
                }
                my_it++;
                right_it++;
            } else if (my_it->first > right_it->first) {
                diff.push_back(COO_diff(COO_diff::LEFT_EMPTY, row_id, right_it->first,
                                        COO_diff::EMPTY_VALUE, right_it->second));
                right_it++;
            } else {
                diff.push_back(COO_diff(COO_diff::RIGHT_EMPTY, row_id, my_it->first, my_it->second,
                                        COO_diff::EMPTY_VALUE));
                my_it++;
            }
        }
    }

    fclose(outf);
}

template <class Element_type>
void CSR_matrix<Element_type>::diff_to_file(const CSR_matrix<Element_type>& right) {
    std::vector<COO_diff> coo_diff;
    diff_to_arr(right, coo_diff);

    FILE *outf = fopen("csr_diff.txt", "w");
    Element_type max_diff = 0.0; int max_diff_row = -1, max_diff_col = -1;
    Element_type max_rela_diff = 0.0; int max_rela_diff_row = -1, max_rela_diff_col = -1;
    for (int i = 0; i < coo_diff.size(); i++) {
        fprintf(outf, "%d %d %d %e %e\n",
                coo_diff[i].m_diff_type, coo_diff[i].m_row, coo_diff[i].m_col,
                coo_diff[i].m_right_val, coo_diff[i].m_left_val);

        if (coo_diff[i].m_diff_type == COO_diff::UNEQUAL) {
            Element_type diff = abs(coo_diff[i].m_right_val - coo_diff[i].m_left_val);
            Element_type rela_diff = abs(coo_diff[i].m_right_val - coo_diff[i].m_left_val) /
                              abs(coo_diff[i].m_right_val + coo_diff[i].m_left_val);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_row = coo_diff[i].m_row;
                max_diff_col = coo_diff[i].m_col;
            }
            if (rela_diff > max_rela_diff) {
                max_rela_diff = rela_diff;
                max_rela_diff_row = coo_diff[i].m_row;
                max_rela_diff_col = coo_diff[i].m_col;
            }
        }
    }

    printf("max_diff = %e, row = %d, col = %d\n", max_diff, max_diff_row, max_diff_col);
    printf("max_rela_diff = %e, row = %d, col = %d\n", max_rela_diff, max_rela_diff_row, max_rela_diff_col);

    fclose(outf);
}

template <class Element_type>
void CSR_matrix<Element_type>::multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec) {
    ASSERT_AND_PRINTF(in_vec.get_size() >= m_num_col && out_vec.get_size() >= m_num_row, "mtx x = %d, in_vec sz = %lu, mtx y = %d, out_vec sz = %lu\n",
                      m_num_col, in_vec.get_size(), m_num_row, out_vec.get_size());

    for (int row = 0; row < m_num_row; row++) {
		Element_type res = 0;

		for (int index = m_row_offsets[row]; index < m_row_offsets[row + 1]; index++)
			res += m_vals[index] * in_vec.at(m_col_idxs[index]/* + m_sub_start_col*/);

		out_vec.at(row/* + m_sub_start_row*/) = res;
	}
}
#if defined(__x86_64__) || defined(__i386__)
template <class Element_type>
sparse_matrix_t* CSR_matrix<Element_type>::convert_to_mkl_matrix() {
    static_assert(std::is_same<MKL_INT, int>::value);

    sparse_matrix_t* ret = new sparse_matrix_t;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_create_csr(ret, SPARSE_INDEX_BASE_ZERO, m_num_row, m_num_col, m_row_offsets, m_row_offsets + 1, m_col_idxs, m_vals);
    } else if (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_create_csr(ret, SPARSE_INDEX_BASE_ZERO, m_num_row, m_num_col, m_row_offsets, m_row_offsets + 1, m_col_idxs, m_vals);
    } else {
        assert(false);
    }
    check_mkl_sparse_ret(mkl_ret);

    return ret;
}

template <class Element_type>
sparse_matrix_t* CSR_matrix<Element_type>::convert_to_mkl_matrix_csc_trans() {
    static_assert(std::is_same<MKL_INT, int>::value);

    sparse_matrix_t* ret = new sparse_matrix_t;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_create_csc(ret, SPARSE_INDEX_BASE_ZERO, m_num_col, m_num_row, m_row_offsets, m_row_offsets + 1, m_col_idxs, m_vals);
    } else if (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_create_csc(ret, SPARSE_INDEX_BASE_ZERO, m_num_col, m_num_row, m_row_offsets, m_row_offsets + 1, m_col_idxs, m_vals);
    } else {
        assert(false);
    }
    check_mkl_sparse_ret(mkl_ret);

    return ret;
}
#endif

template <class Element_type>
CSC_matrix<Element_type>::CSC_matrix(int num_row, int num_col, int sub_start_row, int sub_start_col, int nz_count) :
                        m_num_row(num_row), m_num_col(num_col), m_sub_start_row(sub_start_row), m_sub_start_col(sub_start_col), m_nz_count(nz_count) {
    m_col_offsets = malloc_with_check<int>(m_num_col + 1, MM_DEFAULT_ALIGNMENT);
    m_row_idxs = malloc_with_check<int>(m_nz_count, MM_DEFAULT_ALIGNMENT);
    m_vals = malloc_with_check<Element_type>(m_nz_count, MM_DEFAULT_ALIGNMENT);
}


template <class Element_type>
CSC_matrix<Element_type>::~CSC_matrix() {
    _mm_free(m_row_idxs);
    _mm_free(m_col_offsets);
    _mm_free(m_vals);
}

template <class Element_type>
void CSC_matrix<Element_type>::multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec) {
    ASSERT_AND_PRINTF(in_vec.get_size() >= m_num_col && out_vec.get_size() >= m_num_row, "mtx x = %d, in_vec sz = %lu, mtx y = %d, out_vec sz = %lu\n",
                      m_num_col, in_vec.get_size(), m_num_row, out_vec.get_size());

	memset(&out_vec.at(0), 0, sizeof(Element_type) * m_num_row);

	for (int col = 0; col < m_num_col; col++) {
		for (int offset = m_col_offsets[col]; offset < m_col_offsets[col + 1]; offset++) {
			int row = m_row_idxs[offset];
			Element_type value = m_vals[offset];
			out_vec.at(row) += value * in_vec.at(col);
		}
	}
}

#if defined(__x86_64__) || defined(__i386__)
template <class Element_type>
sparse_matrix_t* CSC_matrix<Element_type>::convert_to_mkl_matrix() {
    static_assert(std::is_same<MKL_INT, int>::value);

    sparse_matrix_t* ret = new sparse_matrix_t;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_create_csc(ret, SPARSE_INDEX_BASE_ZERO, m_num_row, m_num_col, m_col_offsets, m_col_offsets + 1, m_row_idxs, m_vals);
    } else if (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_create_csc(ret, SPARSE_INDEX_BASE_ZERO, m_num_row, m_num_col, m_col_offsets, m_col_offsets + 1, m_row_idxs, m_vals);
    } else {
        assert(false);
    }
    check_mkl_sparse_ret(mkl_ret);

    return ret;
}
template <class Element_type>
sparse_matrix_t* CSC_matrix<Element_type>::convert_to_mkl_matrix_csr_trans() {
    static_assert(std::is_same<MKL_INT, int>::value);

    sparse_matrix_t* ret = new sparse_matrix_t;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_create_csr(ret, SPARSE_INDEX_BASE_ZERO, m_num_col, m_num_row, m_col_offsets, m_col_offsets + 1, m_row_idxs, m_vals);
    } else if (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_create_csr(ret, SPARSE_INDEX_BASE_ZERO, m_num_col, m_num_row, m_col_offsets, m_col_offsets + 1, m_row_idxs, m_vals);
    } else {
        assert(false);
    }
    check_mkl_sparse_ret(mkl_ret);

    return ret;
}
#endif

template <class Element_type>
COO_matrix_buffer<Element_type>::COO_matrix_buffer(int mat_rows, int mat_cols, int sub_start_row, int sub_start_col) {
    m_num_row = mat_rows;
    m_num_col = mat_cols;
    m_sub_start_row = sub_start_row;
    m_sub_start_col = sub_start_col;

    m_nz_count = 0;
    m_row_nz_count.resize(m_num_row);
    m_col_nz_count.resize(m_num_col);
    memset(&m_row_nz_count[0], 0, sizeof(int) * m_num_row);
    memset(&m_col_nz_count[0], 0, sizeof(int) * m_num_col);
}

template <class Element_type>
COO_matrix_buffer<Element_type>::~COO_matrix_buffer() {
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::recount_nz_count() {
    m_row_nz_count.resize(m_num_row);
    m_col_nz_count.resize(m_num_col);
    memset(&m_row_nz_count[0], 0, sizeof(int) * m_num_row);
    memset(&m_col_nz_count[0], 0, sizeof(int) * m_num_col);

    for (int element_id = 0; element_id < m_nz_count; element_id++) {
        m_row_nz_count[m_nz_row_idx[element_id]]++;
        m_col_nz_count[m_nz_col_idx[element_id]]++;
    }
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::add_element(int global_row, int global_col, Element_type val) {
    add_sub_element(global_row - m_sub_start_row, global_col - m_sub_start_col, val);
}

template <class Element_type>
template <class Sub_element_type>
void COO_matrix_buffer<Element_type>::add_sub_element(int sub_row, int sub_col, Sub_element_type val) {
    ASSERT_AND_PRINTF(sub_row >= 0 && sub_row < m_num_row, "sub_row %d not in range %d %d\n", sub_row, 0, m_num_row);
    ASSERT_AND_PRINTF(sub_col >= 0 && sub_col < m_num_col, "sub_col %d not in range %d %d\n", sub_col, 0, m_num_col);
    m_nz_row_idx.push_back(sub_row);
    m_nz_col_idx.push_back(sub_col);
    m_vals.push_back(val);
    m_row_nz_count[sub_row]++;
    m_col_nz_count[sub_col]++;
    m_nz_count++;
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::add_coo_buffer(const COO_matrix_buffer& right) {
    for (int i = 0; i < right.m_vals.size(); i++) {
        // not in the range of this matrix
        int global_row_idx = right.m_nz_row_idx[i] + right.m_sub_start_row;
        int global_col_idx = right.m_nz_col_idx[i] + right.m_sub_start_col;
        if (!check_in_range(global_row_idx, global_col_idx))
            continue;
        add_element(global_row_idx, global_col_idx, right.m_vals[i]);
    }
}

template <class Element_type>
bool COO_matrix_buffer<Element_type>::check_in_range(int global_row_idx, int global_col_idx) {
    if (global_row_idx < m_sub_start_row || global_row_idx >= m_sub_start_row + m_num_row)
        return false;

    if (global_col_idx < m_sub_start_col || global_col_idx >= m_sub_start_col + m_num_col)
        return false;

    return true;
}


template <class Element_type>
template <class Out_element_type>
CSR_matrix<Out_element_type>* COO_matrix_buffer<Element_type>::convert_to_csr_matrix() const {
    CSR_matrix<Out_element_type>* ret = new CSR_matrix<Out_element_type>(m_num_row, m_num_col, m_sub_start_row, m_sub_start_col, m_nz_count);

    int* row_offsets = ret->m_row_offsets;
    int* col_idxs = ret->m_col_idxs;
    Out_element_type* vals = ret->m_vals;

    int* tmp_row_offsets = malloc_with_check<int>(m_num_row + 1);

    // init row offsets
    int cur_nz_count = 0;
    for (int i = 0; i < m_num_row; i++) {
        tmp_row_offsets[i] = row_offsets[i] = cur_nz_count;
        cur_nz_count += m_row_nz_count[i];
    }
    row_offsets[m_num_row] = cur_nz_count;
    assert(cur_nz_count == m_nz_count);

    for (int i = 0; i < m_nz_count; i++) {
        int row = m_nz_row_idx[i];
        int col = m_nz_col_idx[i];
        Element_type val = m_vals[i];

        col_idxs[tmp_row_offsets[row]] = col;
        vals[tmp_row_offsets[row]] = val;

        tmp_row_offsets[row]++;
    }

    for (int i = 0; i < m_num_row; i++) {
        int *row_i_nz_col_index = col_idxs + row_offsets[i];
        Out_element_type *row_i_nz_valus = vals + row_offsets[i];
        int row_i_nz_count = row_offsets[i + 1] - row_offsets[i];

        quick_sort_index_by_value(row_i_nz_col_index, row_i_nz_valus, 0, row_i_nz_count - 1);
    }

    free(tmp_row_offsets);

    return ret;
}

template <class Element_type>
template <class Out_element_type>
CSC_matrix<Out_element_type>* COO_matrix_buffer<Element_type>::convert_to_csc_matrix() const {
    CSC_matrix<Out_element_type>* ret = new CSC_matrix<Out_element_type>(m_num_row, m_num_col, m_sub_start_row, m_sub_start_col, m_nz_count);

    int* col_offsets = ret->m_col_offsets;
    int* row_idxs = ret->m_row_idxs;
    Out_element_type* vals = ret->m_vals;

    int* tmp_col_offsets = malloc_with_check<int>(m_num_col + 1);

    // init col offsets
    int cur_nz_count = 0;
    for (int i = 0; i < m_num_col; i++) {
        tmp_col_offsets[i] = col_offsets[i] = cur_nz_count;
        cur_nz_count += m_col_nz_count[i];
    }
    col_offsets[m_num_col] = cur_nz_count;
    assert(cur_nz_count == m_nz_count);

    for (int i = 0; i < m_nz_count; i++) {
        int col = m_nz_col_idx[i];
        int row = m_nz_row_idx[i];
        Element_type val = m_vals[i];

        row_idxs[tmp_col_offsets[col]] = row;
        vals[tmp_col_offsets[col]] = val;

        tmp_col_offsets[col]++;
    }

    for (int i = 0; i < m_num_col; i++) {
        int *col_i_nz_row_index = row_idxs + col_offsets[i];
        Out_element_type *col_i_nz_valus = vals + col_offsets[i];
        int col_i_nz_count = col_offsets[i + 1] - col_offsets[i];

        quick_sort_index_by_value(col_i_nz_row_index, col_i_nz_valus, 0, col_i_nz_count - 1);
    }

    free(tmp_col_offsets);

    ret->m_flag_row_sorted = true;

    return ret;
}

template <class Element_type>
double COO_matrix_buffer<Element_type>::get_checksum() const {
    double ret = 0.0;
    for (int i = 0; i < m_vals.size(); i++) {
        ret += m_vals[i];
    }
    return ret;
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::resolve(CSR_matrix<Element_type>* mtx) {
    int* row_offsets = mtx->m_row_offsets;
    int* col_idxs = mtx->m_col_idxs;
    Element_type* mtx_vals = (Element_type*) mtx->m_vals;

    // insert elements
    for (int row = 0; row < mtx->m_num_row; row++) {
        for (int index = row_offsets[row]; index < row_offsets[row + 1]; index++)
            add_element(row + mtx->m_sub_start_row, col_idxs[index] + mtx->m_sub_start_col, mtx_vals[index]);
    }
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::resolve(COO_matrix_buffer<Element_type>* mtx) {
    for (int i = 0; i < mtx->m_vals.size(); i++) {
        add_element(mtx->m_nz_row_idx[i] + mtx->m_sub_start_row, mtx->m_nz_col_idx[i] + mtx->m_sub_start_col, mtx->m_vals[i]);
    }
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::multiply_dense_vector(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec) {
    ASSERT_AND_PRINTF(in_vec.get_size() >= m_num_col && out_vec.get_size() >= m_num_row, "mtx x = %d, in_vec sz = %lu, mtx y = %d, out_vec sz = %lu\n",
                      m_num_col, in_vec.get_size(), m_num_row, out_vec.get_size());

    memset(&out_vec.at(0), 0, sizeof(Element_type) * m_num_row);
    for (int i = 0; i < m_vals.size(); i++) {
        out_vec.at(m_nz_row_idx[i]) += m_vals[i] * in_vec.at(m_nz_col_idx[i]);
    }
}

template <class Element_type>
void COO_matrix_buffer<Element_type>::multiply_dense_vector_trans(const Dense_vector<Element_type>& in_vec, const Dense_vector<Element_type>& out_vec) {
    ASSERT_AND_PRINTF(in_vec.get_size() >= m_num_row && out_vec.get_size() >= m_num_col, "mtx y = %d, in_vec sz = %lu, mtx x = %d, out_vec sz = %lu\n",
                      m_num_col, in_vec.get_size(), m_num_row, out_vec.get_size());

    memset(&out_vec.at(0), 0, sizeof(Element_type) * m_num_col);
    for (int i = 0; i < m_vals.size(); i++) {
        out_vec.at(m_nz_col_idx[i]) += m_vals[i] * in_vec.at(m_nz_row_idx[i]);
    }
}

template <class Element_type>
COO_matrix_buffer<Element_type>* COO_matrix_buffer<Element_type>::read_from_file(std::string filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        return nullptr;

    COO_matrix_buffer<Element_type>* ret = new COO_matrix_buffer<Element_type>(1, 1, 1, 1);

    in.read((char*)&ret->m_num_row, sizeof(int));
    in.read((char*)&ret->m_num_col, sizeof(int));
    in.read((char*)&ret->m_sub_start_row, sizeof(int));
    in.read((char*)&ret->m_sub_start_col, sizeof(int));
    in.read((char*)&ret->m_nz_count, sizeof(int));

    ret->m_nz_row_idx.resize(ret->m_nz_count);
    ret->m_nz_col_idx.resize(ret->m_nz_count);
    ret->m_vals.resize(ret->m_nz_count);

    in.read((char*)&ret->m_nz_row_idx.at(0), ret->m_nz_count * sizeof(int));
    in.read((char*)&ret->m_nz_col_idx.at(0), ret->m_nz_count * sizeof(int));
    in.read((char*)&ret->m_vals.at(0), ret->m_nz_count * sizeof(Element_type));

    ret->m_row_nz_count.resize(ret->m_num_row);
    ret->m_col_nz_count.resize(ret->m_num_col);

    in.read((char*)&ret->m_row_nz_count.at(0), ret->m_num_row * sizeof(int));
    in.read((char*)&ret->m_col_nz_count.at(0), ret->m_num_col * sizeof(int));

    in.close();

    return ret;
}

template <class Element_type>
bool COO_matrix_buffer<Element_type>::dump_to_file(std::string filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        return false;

    out.write((char*)&m_num_row, sizeof(int));
    out.write((char*)&m_num_col, sizeof(int));
    out.write((char*)&m_sub_start_row, sizeof(int));
    out.write((char*)&m_sub_start_col, sizeof(int));
    out.write((char*)&m_nz_count, sizeof(int));

    out.write((char*)&m_nz_row_idx.at(0), m_nz_count * sizeof(int));
    out.write((char*)&m_nz_col_idx.at(0), m_nz_count * sizeof(int));
    out.write((char*)&m_vals.at(0), m_nz_count * sizeof(Element_type));

    out.write((char*)&m_row_nz_count.at(0), m_num_row * sizeof(int));
    out.write((char*)&m_col_nz_count.at(0), m_num_col * sizeof(int));

    out.close();

    return true;
}
