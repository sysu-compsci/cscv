#pragma once

#include "arch/pthread_timer.hpp"
#include "arch/ring_fake_barrier.hpp"
#include "cscv/compute.hpp"
#include "cscv/cscv.hpp"
#include "ct/ct_common.hpp"
#include "ct/ct_image.hpp"

enum class Sparse_mkl_flag {
    NO_MKL,
    CSR,
    CSC,
};

template <class Element_type>
struct Tea_soa_context {
    // the copy of cfg
    Computation_config m_comp_cfg;

    // global info
    Img_param m_img_param;
    Dense_vector<Element_type>* m_y_global;
    Image_CT<Element_type>* m_x_input_global;

    Ring_fake_barrier* m_y_reduction_barrier = nullptr;

    // part info
    int m_part_img_x_size, m_part_img_y_size, m_part_angle_count;
    int m_part_start_img_x, m_part_start_img_y, m_part_start_angle;
    int m_part_y_stride;  // usually the global
    int m_part_max_bin_per_angle;  // padded
    Dense_vector<Element_type>* m_part_y;
    Image_CT<Element_type>* m_part_x;

    Dense_vector<int>* m_part_y_lhss = nullptr;
    Dense_vector<int>* m_part_y_rhss = nullptr;

    // block parameter
    uint8_t m_x_group_size, m_y_group_size, m_angle_group_size;
    
    // blocks info
    int m_block_count;

    // naive, no order
    Dense_vector<int>* m_block_x_starts = nullptr;
    Dense_vector<int>* m_block_y_starts = nullptr;
    Dense_vector<int>* m_block_angle_starts = nullptr;
    Dense_vector<int>* m_block_data_size = nullptr;  // tea data evv
    Dense_vector<int>* m_block_bin_count = nullptr;
    Dense_vector<int>* m_block_px_group_count = nullptr;

    Dense_vector<int>* m_block_angle_bin_starts = nullptr;  // block_count * t_vec_angle, even some block will not contains complete t_vec_angle angles
    Dense_vector<int>* m_block_angle_bin_starts_to_lhs = nullptr;  // block_count * t_vec_angle

    // tea
    Dense_vector<Element_type>* m_tea_data = nullptr;
    Dense_vector<uint16_t>* m_tea_pxs = nullptr;  // block len: pxg_count * pxg_size
    Dense_vector<uint8_t>* m_tea_bin_offs = nullptr;  // block len: pxg_count
    Dense_vector<uint16_t>* m_tea_group_offs = nullptr;  // block len: max_bin_count

    // cat
    Dense_vector<Element_type>* m_cat_data = nullptr;
    Dense_vector<std::byte>* m_cat_masks = nullptr;
    Dense_vector<uint8_t>* m_cat_popcnts = nullptr;
    Dense_vector<int>* m_block_cat_data_size = nullptr;  // cat data evv
    Dense_vector<int>* m_block_cat_masks_bytes = nullptr;
    Dense_vector<int>* m_block_cat_popcnts_size = nullptr;

    // block accumulate info
    uint16_t m_block_max_bin;
    uint32_t m_block_max_data;

    std::map<int, uint64_t>* m_tea_y_ax_timers;
    std::map<int, uint64_t>* m_cat_y_ax_timers;

    ~Tea_soa_context() {
        if (m_block_x_starts) delete m_block_x_starts;
        if (m_block_y_starts) delete m_block_y_starts;
        if (m_block_angle_starts) delete m_block_angle_starts;
        if (m_block_data_size) delete m_block_data_size;
        if (m_block_bin_count) delete m_block_bin_count;
        if (m_block_px_group_count) delete m_block_px_group_count;
        if (m_block_angle_bin_starts) delete m_block_angle_bin_starts;
        if (m_tea_data) delete m_tea_data;
        if (m_tea_pxs) delete m_tea_pxs;
        if (m_tea_bin_offs) delete m_tea_bin_offs;
        if (m_tea_group_offs) delete m_tea_group_offs;
        if (m_cat_data) delete m_cat_data;
        if (m_cat_masks) delete m_cat_masks;
        if (m_cat_popcnts) delete m_cat_popcnts;
        if (m_block_cat_data_size) delete m_block_cat_data_size;
        if (m_block_cat_masks_bytes) delete m_block_cat_masks_bytes;
        if (m_block_cat_popcnts_size) delete m_block_cat_popcnts_size;
    }
};

template <class Element_type, size_t t_vec_angle, size_t t_px_group, bool t_use_cat>
void tea_blocks_compute_y_ax(Tea_soa_context<Element_type>* context);

template <class Element_type, size_t t_vec_angle, bool t_use_csc>
void mkl_blocks_compute_y_ax(Tea_soa_context<Element_type>* context);
