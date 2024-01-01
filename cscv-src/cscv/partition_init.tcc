#pragma once

#include "partition.hpp"

template <class Element_type>
double Data_holder<Element_type>::init_full_coo() {
    if (m_generated_full_coo)
        return 0;
    double stt = omp_get_wtime();
    Range_nd<2> img_range;
    img_range[0] = Range_1d(0, m_img_param.m_img_size);
    img_range[1] = Range_1d(0, m_img_param.m_img_size);
    Range_1d angle_range(0, m_img_param.m_num_angle);

    bool read_from_snapshot = false;
    std::string snapshot_filename = strprintf("%s_constant_%d.dat", m_img_param.filename_string().c_str(), m_comp_cfg.m_gen_constant ? 1 : 0);

    if (m_comp_cfg.m_snapshot_for_full) {
        m_coo_full = COO_matrix_buffer<Element_type>::read_from_file(snapshot_filename);
        read_from_snapshot = m_coo_full != nullptr;
    }

    if (!read_from_snapshot) {
        if (!m_comp_cfg.m_gen_constant)
            m_coo_full = m_mtx_generator->generate_system_matrix<Element_type>(img_range, angle_range);
        else
            m_coo_full = m_mtx_generator->generate_system_matrix_constant<Element_type>(img_range, angle_range);

        if (m_comp_cfg.m_snapshot_for_full)
            m_coo_full->dump_to_file(snapshot_filename);
    } else {
        printf("successfully read from snapshot for coo full\n");
    }

    m_generated_full_coo = true;
    m_original_nnz = m_coo_full->m_nz_count;
    m_init_timer_table.collect_data((int)Init_process_type::COO_FULL, omp_get_wtime() - stt);
    return omp_get_wtime() - stt;
}

template <class Element_type>
double Data_holder<Element_type>::init_full_csc() {
    if (m_generated_full_csc)
        return 0;
    mkl_set_num_threads(m_num_threads);
    ASSERT_AND_PRINTF(m_generated_full_coo, "full coo is not generated!\n");
    double stt = omp_get_wtime();
    m_csc_full = m_coo_full->template convert_to_csc_matrix<Element_type>();
    m_csc_mkl_full = m_csc_full->convert_to_mkl_matrix();
    m_csr_trans_mkl_full = m_csc_full->convert_to_mkl_matrix_csr_trans();
    m_generated_full_csc = true;
    m_init_timer_table.collect_data((int)Init_process_type::CSC_FULL, omp_get_wtime() - stt);
    mkl_set_num_threads(1);
    return omp_get_wtime() - stt;
}

template <class Element_type>
double Data_holder<Element_type>::init_full_csr() {
    if (m_generated_full_csr)
        return 0;
    mkl_set_num_threads(m_num_threads);
    ASSERT_AND_PRINTF(m_generated_full_coo, "full coo is not generated!\n");
    double stt = omp_get_wtime();
    m_csr_full = m_coo_full->template convert_to_csr_matrix<Element_type>();
    m_csr_mkl_full = m_csr_full->convert_to_mkl_matrix();
    m_csc_trans_mkl_full = m_csr_full->convert_to_mkl_matrix_csc_trans();
    m_generated_full_csr = true;
    m_init_timer_table.collect_data((int)Init_process_type::CSR_FULL, omp_get_wtime() - stt);
    mkl_set_num_threads(1);
    return omp_get_wtime() - stt;
}

template <class Element_type>
double Data_holder<Element_type>::init_parts_coo() {
    if (m_generated_part_coo)
        return 0;
    double stt = omp_get_wtime();

    Member_func task_func = [](Data_holder<Element_type>* that) {
        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_parts.size(); part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];

            Range_nd<2> img_range;
            img_range[0] = Range_1d(part->m_start_x, part->m_size_image_x);
            img_range[1] = Range_1d(part->m_start_y, part->m_size_image_y);
            Range_1d angle_range = Range_1d(part->m_start_angle, part->m_angle_count);

            if (!that->m_comp_cfg.m_gen_constant)
                part->m_coo_part = that->m_mtx_generator->template generate_system_matrix<Element_type>(img_range, angle_range);
            else
                part->m_coo_part = that->m_mtx_generator->template generate_system_matrix_constant<Element_type>(img_range, angle_range);

            update_coo_bin_range_in_angles(part->m_coo_part, part->m_num_bin, part->m_angle_count, part->m_y_lhss, part->m_y_rhss);
        }
    };

    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(task_func), this);

    m_generated_part_coo = true;
    m_init_timer_table.collect_data((int)Init_process_type::COO_PART, omp_get_wtime() - stt);
    return omp_get_wtime() - stt;
}

template <class Element_type>
double Data_holder<Element_type>::init_blocks_coo_debug_only() {
    double stt = omp_get_wtime();
    Member_func task_func = [](Data_holder<Element_type>* that) {
        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_part_count; part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];

            // the first step: do statistics for blocks
            for (int block_id = 0; block_id < part->m_blocks.size(); block_id++) {
                Block<Element_type>* block = part->m_blocks.at(block_id);
                Range_nd<2> pixel_range;
                pixel_range[0] = Range_1d(block->m_start_x, block->m_size_image_x);
                pixel_range[1] = Range_1d(block->m_start_y, block->m_size_image_y);
                Range_1d angle_range(block->m_start_angle, block->m_angle_count);

                if (!that->m_comp_cfg.m_gen_constant)
                    block->m_coo_block = that->m_mtx_generator->generate_system_matrix<Element_type>(pixel_range, angle_range);
                else
                    block->m_coo_block = that->m_mtx_generator->generate_system_matrix_constant<Element_type>(pixel_range, angle_range);
            }
        }
    };
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(task_func), this);
    return omp_get_wtime() - stt;
}

template <class Element_type>
void Data_holder<Element_type>::free_parts_coo() {
    if (!m_generated_part_coo)
        return;

    for (int part_id = 0; part_id < m_parts.size(); part_id++) {
        Part<Element_type>* part = m_parts[part_id];
        delete part->m_coo_part;
        part->m_coo_part = nullptr;
    }
}

template <class Element_type>
void Data_holder<Element_type>::free_full_coo() {
    if (!m_generated_full_coo)
        return;

    delete m_coo_full;
    m_generated_full_coo = false;
    m_coo_full = nullptr;
}
