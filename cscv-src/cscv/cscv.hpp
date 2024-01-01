#pragma once

#include <set>

#include "base/basic_definition.hpp"
#include "base/string_table.hpp"
#include "cscv/compute.hpp"
#include "ct/ct_image.hpp"
#include "data/data_container.hpp"

enum class Timer_type {
    // common process for part computing, pre-process
    PART_INIT,

    // SoA (y = Ax)
    GET_BLOCK_X,
    SET_YT_0,
    DO_SPMV_Y_AX,
    TRANSPOSE_YT,
    REDUCE_BLOCK_Y,
    TEMP_Y_PROCESS,

    // Other frameworks

    // common process for part computing, post-process
    SPIN,
    REDUCE_PART_X,
    REDUCE_PART_Y,

    // statistics
    SUM,
};

enum class Result_type {
    COO_PART,
    COO_FULL,
    MKL_CSR_FULL,
    MKL_CSC_FULL,
    CSCVB_TEA_BLOCK,
    CSCVB_CAT_BLOCK,
    PART_Y_REDUCTION,
    PART_X_REDUCTION,
};

enum class SPMV_direction {
    Y_AX = 0,
};

static const std::map<Timer_type, std::string> c_timer_str_map {
    {Timer_type::PART_INIT, "PART_INIT"},
    {Timer_type::GET_BLOCK_X, "GET_BLOCK_X"},
    {Timer_type::SET_YT_0, "SET_YT_0"},
    {Timer_type::DO_SPMV_Y_AX, "DO_SPMV_Y_AX"},
    {Timer_type::TRANSPOSE_YT, "TRANS_YT"},
    {Timer_type::REDUCE_BLOCK_Y, "REDUCE_BLOCK_Y"},
    {Timer_type::TEMP_Y_PROCESS, "TEMP_Y_PROCESS"},
    {Timer_type::SPIN, "SPIN"},
    {Timer_type::REDUCE_PART_X, "REDUCE_PART_X"},
    {Timer_type::REDUCE_PART_Y, "REDUCE_PART_Y"},
    {Timer_type::SUM, "SUM"},
};

static const std::map<Result_type, std::string> c_result_label_map {
    {Result_type::COO_PART, "COO_PART"},
    {Result_type::COO_FULL, "COO_FULL"},
    {Result_type::MKL_CSR_FULL, "MKL_CSR_FULL"},
    {Result_type::MKL_CSC_FULL, "MKL_CSC_FULL"},
    {Result_type::CSCVB_TEA_BLOCK, "CSCVB_TEA_BLOCK"},
    {Result_type::CSCVB_CAT_BLOCK, "CSCVB_CAT_BLOCK"},
    {Result_type::PART_X_REDUCTION, "PART_X_REDUCTION"},
    {Result_type::PART_Y_REDUCTION, "PART_Y_REDUCTION"},
};

template <class Element_type>
struct CSCVB_matrix_block {
    int m_offset_for_pixel;  // = m_max_nnz_bin_in_angle_for_pixel * m_angle_count
    int m_size_image_x, m_size_image_y;
    int m_block_angle_bin_count;  // the rhs - lhs + 1 of ranges for all px
    int m_angle_group_size;

    Dense_vector<int>* m_angle_bin_starts = nullptr;  // size: angle count

    Dense_vector<int>* m_pixel_bin_count = nullptr;  // size: px_count. the nnz bin for each px

    int m_pxg_count, m_pxg_size;

    // tea
    Dense_vector<Element_type>* m_tea_data = nullptr;
    uint16_t m_tea_group_offs[c_pxg_max_bin + 1];
    Dense_vector<uint16_t>* m_tea_pxs = nullptr;
    Dense_vector<uint8_t>* m_tea_bin_offs = nullptr;

    // cat
    Dense_vector<Element_type>* m_cat_data = nullptr;
    Dense_vector<std::byte>* m_cat_masks = nullptr;
    Dense_vector<uint8_t>* m_cat_popcnts = nullptr;

    // only used for the non-SoA version
    Dense_vector<Element_type>* m_yt_buffer = nullptr;

    ~CSCVB_matrix_block();
};

struct Computation_config {
    int m_loop_count;

    bool m_gen_constant;

    bool m_run_mkl_csr_full, m_run_mkl_csc_full, m_run_coo_full;
    bool m_run_cscvb_tea, m_run_cscvb_cat;

    bool m_run_normal;

    bool m_print_thread_imba;

    bool m_disable_validation;

    int m_stack_size_mb;

    bool m_print_cscv_expansion_only;
    int m_ref_px_x_offset, m_ref_px_y_offset;

    int m_mempool_mb;

    bool m_print_block_arr_size;

    bool m_validate_generation_only;

    bool m_snapshot_for_full;
};

#include "cscvb.tcc"
