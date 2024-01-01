#pragma once

#include "base/basic_definition.hpp"
#include "ct_common.hpp"
#include "data/data_container.hpp"

/**
 * System_matrix_generator
 *  Description:
 *      Generate system matrix from an Img_param and the given range.
 **/
class System_matrix_generator {
    Img_param m_img_param;

public:
    System_matrix_generator(const Img_param& img_param) : m_img_param(img_param) {}
    // Now, only generate single precision system matrix.
    template <class Element_type = float>
    COO_matrix_buffer<Element_type>* generate_system_matrix(Range_nd<2> img_range, Range_1d angle_range);
    template <class Element_type = float>
    COO_matrix_buffer<Element_type>* generate_system_matrix_constant(Range_nd<2> img_range, Range_1d angle_range);

    void set_img_param(const Img_param& img_param) { m_img_param = m_img_param; }
    const Img_param& get_img_param() const { return m_img_param; }
};

typedef void (*Matrix_fill_func)(void *target, int sub_row_id, int sub_col_id, float value);

void generate_sub_system_matrix_linear_inner(Img_param img_param, Range_nd<2> img_range, Range_1d angle_range,
                                             void *fill_target, Matrix_fill_func fill_func);
void generate_sub_system_matrix_constant_inner(Img_param img_param, Range_nd<2> img_range, Range_1d angle_range,
                                               void *fill_target, Matrix_fill_func fill_func);

template <class Element_type>
COO_matrix_buffer<Element_type>* System_matrix_generator::generate_system_matrix(Range_nd<2> img_range, Range_1d angle_range) {
    COO_matrix_buffer<Element_type>* ret = new COO_matrix_buffer<Element_type>(angle_range.m_size * m_img_param.m_num_bin,
                                                                               img_range.m_ranges[0].m_size * img_range.m_ranges[1].m_size, 0, 0);

    generate_sub_system_matrix_linear_inner(m_img_param, img_range, angle_range, ret,
                                            cast_pointer<Matrix_fill_func>(&COO_matrix_buffer<Element_type>::template add_sub_element<float>));

    return ret;
}

template <class Element_type>
COO_matrix_buffer<Element_type>* System_matrix_generator::generate_system_matrix_constant(Range_nd<2> img_range, Range_1d angle_range) {
    COO_matrix_buffer<Element_type>* ret = new COO_matrix_buffer<Element_type>(angle_range.m_size * m_img_param.m_num_bin,
                                                                               img_range.m_ranges[0].m_size * img_range.m_ranges[1].m_size, 0, 0);

    generate_sub_system_matrix_constant_inner(m_img_param, img_range, angle_range, ret,
                                              cast_pointer<Matrix_fill_func>(&COO_matrix_buffer<Element_type>::template add_sub_element<float>));

    return ret;
}

template <class Element_type>
void update_coo_bin_range_in_angles(COO_matrix_buffer<Element_type>* coo, int num_bin, int angle_count,
                                 Dense_vector<int>* lhss, Dense_vector<int>* rhss) {
    ASSERT_AND_PRINTF(lhss->get_size() >= angle_count, "");
    ASSERT_AND_PRINTF(rhss->get_size() >= angle_count, "");

    for (int ele_id = 0; ele_id < coo->m_nz_count; ele_id++) {
        int angle = coo->m_nz_row_idx[ele_id] / num_bin;
        int bin = coo->m_nz_row_idx[ele_id] % num_bin;

        lhss->at(angle) = std::min(lhss->at(angle), bin);
        rhss->at(angle) = std::max(rhss->at(angle), bin);
    }
}

void free_system_matrix_gen_tmp_data();
