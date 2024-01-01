#pragma once

#include "partition.hpp"

template <class Element_type>
double Data_holder<Element_type>::compute_full_coo_y_ax() {
    ASSERT_AND_PRINTF(m_generated_full_coo, "full coo is not generated!\n");
    m_current_result_type = Result_type::COO_FULL;

    auto stt = std::chrono::high_resolution_clock::now();

    m_coo_full->multiply_dense_vector(*m_x_input->m_vec, *m_y_tmp);

    auto edt = std::chrono::high_resolution_clock::now();

    if (m_y_result_table.get_result_size((int)Result_type::COO_FULL) == 0) {
        m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::COO_FULL, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
    }
    m_calc_y_timer_table.collect_data((int)Result_type::COO_FULL, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9;
}

template <class Element_type>
double Data_holder<Element_type>::compute_full_csc_mkl_y_ax() {
    ASSERT_AND_PRINTF(m_generated_full_csc, "full csc is not generated!\n");
    m_current_result_type = Result_type::MKL_CSC_FULL;

    mkl_set_num_threads(m_num_threads);
    ASSERT_AND_PRINTF(m_num_threads == mkl_get_max_threads(), "thread count %d != %d\n", m_num_threads, mkl_get_max_threads());
    auto stt = std::chrono::high_resolution_clock::now();

    matrix_descr mtx_descr;
    mtx_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *m_csc_mkl_full, mtx_descr,
                                  &(m_x_input->m_vec->at(0)), 0.0, &(m_y_tmp->at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else if constexpr (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *m_csc_mkl_full, mtx_descr,
                                  &(m_x_input->m_vec->at(0)), 0.0, &(m_y_tmp->at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else {
        assert(false);
    }

    auto edt = std::chrono::high_resolution_clock::now();
    mkl_set_num_threads(1);

    if (m_y_result_table.get_result_size((int)Result_type::MKL_CSC_FULL) == 0) {
        m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::MKL_CSC_FULL, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
    }
    m_calc_y_timer_table.collect_data((int)Result_type::MKL_CSC_FULL, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9;
}

template <class Element_type>
double Data_holder<Element_type>::compute_full_csr_mkl_y_ax() {
    ASSERT_AND_PRINTF(m_generated_full_csr, "full csr is not generated!\n");
    m_current_result_type = Result_type::MKL_CSR_FULL;

    mkl_set_num_threads(m_num_threads);
    ASSERT_AND_PRINTF(m_num_threads == mkl_get_max_threads(), "thread count %d != %d\n", m_num_threads, mkl_get_max_threads());
    auto stt = std::chrono::high_resolution_clock::now();

    matrix_descr mtx_descr;
    mtx_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *m_csr_mkl_full, mtx_descr,
                                  &(m_x_input->m_vec->at(0)), 0.0, &(m_y_tmp->at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else if constexpr (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *m_csr_mkl_full, mtx_descr,
                                  &(m_x_input->m_vec->at(0)), 0.0, &(m_y_tmp->at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else {
        assert(false);
    }

    auto edt = std::chrono::high_resolution_clock::now();
    mkl_set_num_threads(1);

    if (m_y_result_table.get_result_size((int)Result_type::MKL_CSR_FULL) == 0) {
        m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::MKL_CSR_FULL, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
    }
    m_calc_y_timer_table.collect_data((int)Result_type::MKL_CSR_FULL, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9;
}

template <class Element_type>
double Data_holder<Element_type>::compute_parts_coo_y_ax() {
    ASSERT_AND_PRINTF(m_generated_part_coo, "part coo is not generated!\n");
    m_current_result_type = Result_type::COO_PART;

    auto stt = std::chrono::high_resolution_clock::now();
    Member_func task_func = [](Data_holder<Element_type>* that) {
        auto calculation_stt = std::chrono::high_resolution_clock::now();
        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_part_count; part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];
            if (part->is_empty())
                continue;

            part->fetch_x(that->m_x_input);

            part->m_coo_part->multiply_dense_vector(*part->m_part_image->m_vec, *part->m_part_detector);
        }
        auto calculation_edt = std::chrono::high_resolution_clock::now();

        #pragma omp barrier

        auto wait_edt = std::chrono::high_resolution_clock::now();

        that->m_thread_calculation_timers[omp_get_thread_num()][Result_type::COO_PART].m_calculation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(calculation_edt - calculation_stt).count() / 1e9;
        that->m_thread_calculation_timers[omp_get_thread_num()][Result_type::COO_PART].m_spin_time += std::chrono::duration_cast<std::chrono::nanoseconds>(wait_edt - calculation_edt).count() / 1e9;
    };
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(task_func), this);
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::reduce_y_tmp_from_parts), this);

    auto edt = std::chrono::high_resolution_clock::now();

    if (m_y_result_table.get_result_size((int)Result_type::COO_PART) == 0) {
        // memcpy(&m_y_std->at(0), &m_y_tmp->at(0), sizeof(Element_type) * m_img_param.m_num_angle * m_img_param.m_num_bin);        
        for (int i = 0; i < m_img_param.m_num_angle * m_img_param.m_num_bin; i++) {
            m_y_std->at(i) = (i % 1000) * (1.0 / 1000) + (i % 1000000) * (1.0 / 1000000);
        }

        m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::COO_PART, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
        m_std_y_produced = true;
    }
    m_calc_y_timer_table.collect_data((int)Result_type::COO_PART, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9;
}
