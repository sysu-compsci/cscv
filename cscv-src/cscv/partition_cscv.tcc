#pragma once

#include "partition.hpp"

#include "cscv/compute.hpp"

template <class Element_type>
double Data_holder<Element_type>::init_blocks_cscvb() {
    if (m_generated_block_cscvb)
        return 0;
    m_cscv_converter->clear_statictics();
    auto stt = std::chrono::high_resolution_clock::now();
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
                    block->m_coo_block = that->m_mtx_generator->template generate_system_matrix<Element_type>(pixel_range, angle_range);
                else
                    block->m_coo_block = that->m_mtx_generator->template generate_system_matrix_constant<Element_type>(pixel_range, angle_range);

                Dense_vector<int> tmp_lhss(block->m_angle_count, &part->m_y_lhss->at(block->m_start_angle - part->m_start_angle));
                Dense_vector<int> tmp_rhss(block->m_angle_count, &part->m_y_rhss->at(block->m_start_angle - part->m_start_angle));

                update_coo_bin_range_in_angles(block->m_coo_block, part->m_num_bin, block->m_angle_count, &tmp_lhss, &tmp_rhss);

                block->m_csc_block = block->m_coo_block->convert_to_csc_matrix();
                delete block->m_coo_block;
                block->m_coo_block = nullptr;
            }
        }

        #pragma omp barrier
        auto stt = std::chrono::high_resolution_clock::now();

        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_part_count; part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];
            for (int block_id = 0; block_id < part->m_blocks.size(); block_id++) {
                Block<Element_type>* block = part->m_blocks.at(block_id);
                block->m_cscv_sta = that->m_cscv_converter->get_statistics_before_convertion(block->m_csc_block, block);
            }
        }
        #pragma omp barrier
        auto sta_wtime = std::chrono::high_resolution_clock::now();

        if (that->m_comp_cfg.m_print_cscv_expansion_only) {
            #pragma omp master
            {
                // print the ref px offset, and the cscv expansion
                printf("[PRINT_CSCV_EXPANSION][CSCV_EXPANSION] %f\n", that->m_cscv_converter->get_cscv_expansion());
                printf("[PRINT_CSCV_EXPANSION][REF_PX_X_OFFSET] %d\n", that->m_comp_cfg.m_ref_px_x_offset);
                printf("[PRINT_CSCV_EXPANSION][REF_PX_Y_OFFSET] %d\n", that->m_comp_cfg.m_ref_px_y_offset);

                exit(0);
            }

            #pragma omp barrier
        }

        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_part_count; part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];
            part->m_cscv_mem_pool = new CSCV_block_mem_pool<Element_type>(part->get_cscv_statistics_in_blocks());

            for (int block_id = 0; block_id < part->m_blocks.size(); block_id++) {
                Block<Element_type>* block = part->m_blocks.at(block_id);

                that->m_cscv_converter->convert_system_matrix_to_cscvb(block->m_csc_block, block, part->m_cscv_mem_pool, block->m_cscv_sta);
            }
        }

        #pragma omp barrier
        auto edt = std::chrono::high_resolution_clock::now();
        if (omp_get_thread_num() == 0) {
            printf("[CONVERTION TIMER] CSCV Sta time = %f, CSC -> CSCV time = %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(sta_wtime - stt).count() / 1e9, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9);
        }

        #pragma omp for nowait
        for (int part_id = 0; part_id < that->m_part_count; part_id++) {
            Part<Element_type>* part = that->m_parts[part_id];
            for (int block_id = 0; block_id < part->m_blocks.size(); block_id++) {
                Block<Element_type>* block = part->m_blocks.at(block_id);
                delete block->m_csc_block;
                block->m_csc_block = nullptr;
            }

            for (int block_id = 0; block_id < part->m_blocks.size(); block_id++) {
                Block<Element_type>* block = part->m_blocks.at(block_id);

                block->m_cscv_sta->free_before_computation();
            }

            part->build_cscvb_yt_buffer();
            part->build_tea_soa();

            if (!that->m_comp_cfg.m_run_cscvb_tea)
                part->m_cscv_mem_pool->free_tea();
            if (!that->m_comp_cfg.m_run_cscvb_cat)
                part->m_cscv_mem_pool->free_cat();
        }
    };
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(task_func), this);

    m_generated_block_cscvb = true;

    m_cscv_converter->get_nnz_expansion_ratio();
    m_init_timer_table.collect_data((int)Init_process_type::CSCVB_BLOCK, (std::chrono::high_resolution_clock::now() - stt).count() / 1e9);

    PRINTF("m_cscv_converter->get_max_pxg_bin_count() = %d, c_pxg_max_bin = %d\n", m_cscv_converter->get_max_pxg_bin_count(), c_pxg_max_bin);
    PRINTF("get_max_cscv_block_bin_count() = %d, c_block_max_bin = %d\n", get_max_cscv_block_bin_count(), c_block_max_bin);
    // assert(c_block_max_bin >= get_max_cscv_block_bin_count());
    // ASSERT_AND_PRINTF(m_cscv_converter->get_max_pxg_bin_count() <= c_pxg_max_bin, "bin count excceed! %d > %d\n",
    //                   m_cscv_converter->get_max_pxg_bin_count(), c_pxg_max_bin);

    return (std::chrono::high_resolution_clock::now() - stt).count() / 1e9;
}

// building soa for tea algorithm
template <class Element_type>
void Part<Element_type>::build_tea_soa() {
    m_tea_soa_context = new Tea_soa_context<Element_type>;
    Tea_soa_context<Element_type>* context = m_tea_soa_context;

    m_tea_soa_context->m_comp_cfg = m_comp_cfg;

    m_tea_soa_context->m_y_reduction_barrier = m_y_reduction_barrier;

    context->m_img_param = m_img_param;
    context->m_y_global = m_y_tmp;
    context->m_x_input_global = m_x_input;

    context->m_part_img_x_size = m_size_image_x;
    context->m_part_img_y_size = m_size_image_x;
    context->m_part_angle_count = m_angle_count;

    context->m_part_start_img_x = m_start_x;
    context->m_part_start_img_y = m_start_y;
    context->m_part_start_angle = m_start_angle;

    context->m_part_y_stride = m_num_bin;

    context->m_part_x = m_part_image;
    context->m_part_y = m_part_detector;

    context->m_part_y_lhss = m_y_lhss;
    context->m_part_y_rhss = m_y_rhss;

    context->m_angle_group_size = m_angle_group_size;
    context->m_x_group_size = m_x_group_size;
    context->m_y_group_size = m_y_group_size;

    context->m_block_count = m_blocks.size();

    context->m_block_max_bin = 0;
    context->m_block_max_data = 0;

    context->m_part_max_bin_per_angle = 0;
    for (int i = 0; i < m_y_lhss->get_size(); i++) {
        context->m_part_max_bin_per_angle = std::max(m_y_rhss->at(i) - m_y_lhss->at(i), context->m_part_max_bin_per_angle);
    }

    context->m_part_max_bin_per_angle = div_and_ceil(context->m_part_max_bin_per_angle, m_angle_group_size) * m_angle_group_size;

    bool tea_or_cat = m_comp_cfg.m_run_cscvb_tea || m_comp_cfg.m_run_cscvb_cat;

    if (context->m_block_count > 0) {
        context->m_block_x_starts = new Dense_vector<int>(context->m_block_count);
        context->m_block_y_starts = new Dense_vector<int>(context->m_block_count);
        context->m_block_angle_starts = new Dense_vector<int>(context->m_block_count);
        context->m_block_data_size = new Dense_vector<int>(context->m_block_count);
        context->m_block_bin_count = new Dense_vector<int>(context->m_block_count);
        context->m_block_px_group_count = new Dense_vector<int>(context->m_block_count);

        context->m_block_angle_bin_starts = new Dense_vector<int>(context->m_block_count * context->m_angle_group_size);
        context->m_block_angle_bin_starts->set_zero();
        context->m_block_angle_bin_starts_to_lhs = new Dense_vector<int>(context->m_block_count * context->m_angle_group_size);
        context->m_block_angle_bin_starts_to_lhs->set_zero();

        context->m_tea_group_offs = new Dense_vector<uint16_t>(context->m_block_count * (c_pxg_max_bin + 1));

        context->m_block_cat_data_size = new Dense_vector<int>(context->m_block_count);
        context->m_block_cat_masks_bytes = new Dense_vector<int>(context->m_block_count);
        context->m_block_cat_popcnts_size = new Dense_vector<int>(context->m_block_count);

        for (int block_id = 0; block_id < m_blocks.size(); block_id++) {
            Block<Element_type>* block = m_blocks.at(block_id);
            CSCVB_matrix_block<Element_type>* cscvb = block->m_cscvb_block;

            if (tea_or_cat) {
                if (block_id != 0) {
                    CSCVB_matrix_block<Element_type>* last_cscvb = m_blocks.at(block_id - 1)->m_cscvb_block;

                    ASSERT_AND_PRINTF(&last_cscvb->m_tea_data->at(0) + last_cscvb->m_tea_data->get_size() == &cscvb->m_tea_data->at(0), "");
                    ASSERT_AND_PRINTF(&last_cscvb->m_tea_pxs->at(0) + last_cscvb->m_tea_pxs->get_size() == &cscvb->m_tea_pxs->at(0), "");
                    ASSERT_AND_PRINTF(&last_cscvb->m_tea_bin_offs->at(0) + last_cscvb->m_tea_bin_offs->get_size() == &cscvb->m_tea_bin_offs->at(0), "");

                    ASSERT_AND_PRINTF(&last_cscvb->m_cat_data->at(0) + last_cscvb->m_cat_data->get_size() == &cscvb->m_cat_data->at(0), "");
                    ASSERT_AND_PRINTF(&last_cscvb->m_cat_masks->at(0) + last_cscvb->m_cat_masks->get_size() == &cscvb->m_cat_masks->at(0), "");
                    ASSERT_AND_PRINTF(&last_cscvb->m_cat_popcnts->at(0) + last_cscvb->m_cat_popcnts->get_size() == &cscvb->m_cat_popcnts->at(0), "");
                }
                context->m_block_max_data = std::max(cscvb->m_tea_data->get_size(), (size_t)context->m_block_max_data);
                context->m_block_data_size->at(block_id) = cscvb->m_tea_data->get_size();
                context->m_block_px_group_count->at(block_id) = cscvb->m_tea_bin_offs->get_size();
                for (int pxg_bin_count = 0; pxg_bin_count < (c_pxg_max_bin + 1); pxg_bin_count++) {
                    context->m_tea_group_offs->at(block_id * (c_pxg_max_bin + 1) + pxg_bin_count) = cscvb->m_tea_group_offs[pxg_bin_count];
                }

                context->m_block_cat_data_size->at(block_id) = cscvb->m_cat_data->get_size();
                context->m_block_cat_masks_bytes->at(block_id) = cscvb->m_cat_masks->get_bytes();
                context->m_block_cat_popcnts_size->at(block_id) = cscvb->m_cat_popcnts->get_bytes();
            }

            context->m_block_max_bin = std::max(cscvb->m_block_angle_bin_count, (int)context->m_block_max_bin);

            context->m_block_x_starts->at(block_id) = block->m_start_x;
            context->m_block_y_starts->at(block_id) = block->m_start_y;
            context->m_block_angle_starts->at(block_id) = block->m_start_angle;
            context->m_block_bin_count->at(block_id) = cscvb->m_block_angle_bin_count;

            for (int block_angle = 0; block_angle < block->m_angle_count; block_angle++) {
                context->m_block_angle_bin_starts->at(block_id * context->m_angle_group_size + block_angle) = cscvb->m_angle_bin_starts->at(block_angle);

                int part_angle = block->m_start_angle - m_start_angle + block_angle;
                context->m_block_angle_bin_starts_to_lhs->at(block_id * context->m_angle_group_size + block_angle) =
                    context->m_block_angle_bin_starts->at(block_id * context->m_angle_group_size + block_angle) - m_y_lhss->at(part_angle);
            }
        }

        if (tea_or_cat) {
            context->m_tea_data = new Dense_vector<Element_type>(m_cscv_mem_pool->get_tea_data_size(), &m_blocks[0]->m_cscvb_block->m_tea_data->at(0));
            context->m_tea_pxs = new Dense_vector<uint16_t>(m_cscv_mem_pool->get_tea_pxs_size(), &m_blocks[0]->m_cscvb_block->m_tea_pxs->at(0));
            context->m_tea_bin_offs = new Dense_vector<uint8_t>(m_cscv_mem_pool->get_tea_bin_offs_size(), &m_blocks[0]->m_cscvb_block->m_tea_bin_offs->at(0));

            context->m_cat_data = new Dense_vector<Element_type>(m_cscv_mem_pool->get_cat_data_size(), &m_blocks[0]->m_cscvb_block->m_cat_data->at(0));
            context->m_cat_masks = new Dense_vector<std::byte>(m_cscv_mem_pool->get_cat_masks_bytes(), &m_blocks[0]->m_cscvb_block->m_cat_masks->at(0));
            context->m_cat_popcnts = new Dense_vector<uint8_t>(m_cscv_mem_pool->get_cat_popcnts_size(), &m_blocks[0]->m_cscvb_block->m_cat_popcnts->at(0));
        }

        auto tl_map = Threads_timer_map::get_local();
        context->m_tea_y_ax_timers = &tl_map->m_timers[(int)SPMV_direction::Y_AX][(int)Result_type::CSCVB_TEA_BLOCK];
        context->m_cat_y_ax_timers = &tl_map->m_timers[(int)SPMV_direction::Y_AX][(int)Result_type::CSCVB_CAT_BLOCK];
    }
}

template <class Element_type>
template <uint8_t t_vec_angle, uint8_t t_px_group>
void Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax_inner() {
    int part_count = 0;

    uint64_t& part_init_time_ref = Threads_timer_map::get_local()->get_time_ref((int)SPMV_direction::Y_AX, (int)m_current_result_type, (int)Timer_type::PART_INIT);
    uint64_t& spin_time_ref = Threads_timer_map::get_local()->get_time_ref((int)SPMV_direction::Y_AX, (int)m_current_result_type, (int)Timer_type::SPIN);
    auto calculation_stt = std::chrono::high_resolution_clock::now();

    // m_y_tmp->set_zero_omp_for();

    #pragma omp for nowait
    for (int part_id = 0; part_id < m_part_count; part_id++) {
        part_count++;
        Part<Element_type>* part = m_parts[part_id];

        rdtsc_interval();
        // part->m_part_detector->set_zero();  // remove this if reduce y in SoA
        part->fetch_x(m_x_input);  // fetch to part local
        part_init_time_ref += rdtsc_interval();

        if (m_use_cat) {
            tea_blocks_compute_y_ax<Element_type, t_vec_angle, t_px_group, true>(part->m_tea_soa_context);
        } else {
            tea_blocks_compute_y_ax<Element_type, t_vec_angle, t_px_group, false>(part->m_tea_soa_context);
        }
    }
    auto calculation_edt = std::chrono::high_resolution_clock::now();
    rdtsc_interval();

    #pragma omp barrier

    spin_time_ref += rdtsc_interval();
    auto wait_edt = std::chrono::high_resolution_clock::now();

    if (m_use_cat) {
        m_thread_calculation_timers[omp_get_thread_num()][Result_type::CSCVB_CAT_BLOCK].m_calculation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(calculation_edt - calculation_stt).count() / 1e9;
        m_thread_calculation_timers[omp_get_thread_num()][Result_type::CSCVB_CAT_BLOCK].m_spin_time += std::chrono::duration_cast<std::chrono::nanoseconds>(wait_edt - calculation_edt).count() / 1e9;
    } else {
        m_thread_calculation_timers[omp_get_thread_num()][Result_type::CSCVB_TEA_BLOCK].m_calculation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(calculation_edt - calculation_stt).count() / 1e9;
        m_thread_calculation_timers[omp_get_thread_num()][Result_type::CSCVB_TEA_BLOCK].m_spin_time += std::chrono::duration_cast<std::chrono::nanoseconds>(wait_edt - calculation_edt).count() / 1e9;
    }
}

template <class Element_type>
template <uint8_t t_vec_angle>
void Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax_px_group_expand() {
    if (m_partition_result.m_pxg_size == 1)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 1>();
    else if (m_partition_result.m_pxg_size == 2)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 2>();
    else if (m_partition_result.m_pxg_size == 4)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 4>();
    else if (m_partition_result.m_pxg_size == 8)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 8>();
    else if (m_partition_result.m_pxg_size == 16)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 16>();
    else if (m_partition_result.m_pxg_size == 32)
        compute_blocks_cscvb_tea_y_ax_inner<t_vec_angle, 32>();
    else
        ASSERT_AND_PRINTF(false, "block kernel not found, angle group = %d\n", m_partition_result.m_angle_group_size);
}

template <class Element_type>
double Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax(bool use_cat) {
    ASSERT_AND_PRINTF(m_generated_block_cscvb, "block cscvb is not generated!\n");
    m_current_result_type = use_cat ? Result_type::CSCVB_CAT_BLOCK : Result_type::CSCVB_TEA_BLOCK;

    m_use_cat = use_cat;

    auto stt = std::chrono::high_resolution_clock::now();

    if (m_partition_result.m_angle_group_size == 4)
        m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax_px_group_expand<4>), this);
    else if (m_partition_result.m_angle_group_size == 8)
        m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax_px_group_expand<8>), this);
    else if (m_partition_result.m_angle_group_size == 16)
        m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::compute_blocks_cscvb_tea_y_ax_px_group_expand<16>), this);
    else
        ASSERT_AND_PRINTF(false, "block kernel not found, px group = %d\n", m_partition_result.m_angle_group_size);

    auto red_stt = std::chrono::high_resolution_clock::now();

    // m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::reduce_y_tmp_from_parts), this);

    auto edt = std::chrono::high_resolution_clock::now();

    if (use_cat) {
        if (m_y_result_table.get_result_size((int)Result_type::CSCVB_CAT_BLOCK) == 0) {
            m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::CSCVB_CAT_BLOCK, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
        }
        m_calc_y_timer_table.collect_data((int)Result_type::CSCVB_CAT_BLOCK, std::chrono::duration_cast<std::chrono::nanoseconds>(red_stt - stt).count() / 1e9);
    } else {
        if (m_y_result_table.get_result_size((int)Result_type::CSCVB_TEA_BLOCK) == 0) {
            m_y_result_table.collect_data_arr_and_memset_0((int)Result_type::CSCVB_TEA_BLOCK, m_y_tmp->get_size(), &(m_y_tmp->at(0)));
        }
        m_calc_y_timer_table.collect_data((int)Result_type::CSCVB_TEA_BLOCK, std::chrono::duration_cast<std::chrono::nanoseconds>(red_stt - stt).count() / 1e9);
    }

    m_calc_y_timer_table.collect_data((int)Result_type::PART_Y_REDUCTION, std::chrono::duration_cast<std::chrono::nanoseconds>(edt - red_stt).count() / 1e9);

    return std::chrono::duration_cast<std::chrono::nanoseconds>(edt - stt).count() / 1e9;
}

template <class Element_type>
void Data_holder<Element_type>::estimate_cscvb_tea_cat_mem_expense() {
    ASSERT_AND_PRINTF(m_generated_block_cscvb, "block cscvb is not generated!\n");

    uint64_t pxg_expense = 0;
    uint64_t mask_expense = 0;

    uint64_t x_byte = 0, y_byte = 0;
    int full_block_count = 0;

    x_byte = m_x_tmp->m_vec->get_bytes();
    y_byte = m_y_tmp->get_bytes();

    for (Part<Element_type>* part : m_parts) {
        for (Block<Element_type>* block : part->m_blocks) {
            pxg_expense += block->m_cscvb_block->m_tea_pxs->get_bytes() + block->m_cscvb_block->m_tea_bin_offs->get_bytes();  // the bin_off arr is ommitted
            mask_expense += block->m_cscvb_block->m_cat_masks->get_bytes();

            full_block_count++;
        }
    }
    // for paper
    uint64_t original_nnz = m_cscv_converter->get_original_nnz();
    uint64_t tea_nnz = m_cscv_converter->get_tea_nnz();
    uint64_t xy_nnz = m_x_tmp->m_vec->get_size() + m_y_tmp->get_size();

    uint64_t tea_float_expense = (tea_nnz + xy_nnz) * 4 + pxg_expense;
    uint64_t tea_double_expense = (tea_nnz + xy_nnz) * 8 + pxg_expense;
    uint64_t cat_float_expense = (original_nnz + xy_nnz) * 4 + pxg_expense + mask_expense;
    uint64_t cat_double_expense = (original_nnz + xy_nnz) * 8 + pxg_expense + mask_expense;

    Logger::get_instance().set_filename(get_workload_filename());
    Logger::get_instance().write(strprintf("[TEA]\n"));

    Logger::get_instance().write(strprintf("tea_nnz\t%lu\n", tea_nnz));
    Logger::get_instance().write(strprintf("pxg_expense\t%lu\n", pxg_expense));
    Logger::get_instance().write(strprintf("mask_expense\t%lu\n", mask_expense));
    Logger::get_instance().write(strprintf("tea_float_expense\t%lu\n", tea_float_expense));
    Logger::get_instance().write(strprintf("tea_double_expense\t%lu\n", tea_double_expense));
    Logger::get_instance().write(strprintf("cat_float_expense\t%lu\n", cat_float_expense));
    Logger::get_instance().write(strprintf("cat_double_expense\t%lu\n", cat_double_expense));

    if (std::is_same<float, Element_type>::value) {
        m_tea_iter_bytes_expense = tea_float_expense;
        m_cat_iter_bytes_expense = cat_float_expense;
    } else if (std::is_same<double, Element_type>::value) {
        m_tea_iter_bytes_expense = tea_double_expense;
        m_cat_iter_bytes_expense = cat_double_expense;
    } else {
        assert(false);
    }

    String_table tea_cat_table("cscvb tea cat expense");

    tea_cat_table.set_value("tea (GB)", "data", tea_nnz * sizeof(Element_type) / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("tea (GB)", "pxg", pxg_expense / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("tea (GB)", "mask", 0);
    tea_cat_table.set_value("tea (GB)", "x & y", xy_nnz * sizeof(Element_type) / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("tea (GB)", "total", m_tea_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("cat (GB)", "data", original_nnz * sizeof(Element_type) / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("cat (GB)", "pxg", pxg_expense / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("cat (GB)", "mask", mask_expense);
    tea_cat_table.set_value("cat (GB)", "x & y", xy_nnz * sizeof(Element_type) / (1024.0 * 1024.0 * 1024.0));
    tea_cat_table.set_value("cat (GB)", "total", m_cat_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0));

    printf("%s", tea_cat_table.to_string().c_str());

    printf("[expense]: tea: %f, cat: %f\n", m_tea_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0), m_cat_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0));
}
