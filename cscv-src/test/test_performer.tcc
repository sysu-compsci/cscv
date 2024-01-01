#pragma once

#include "test_performer.hpp"

template <class Element_type>
void Test_performer::run_inner() {
    Img_param img_param(m_img_size, 0.0, m_delta_angle, m_num_angle, m_num_bin, 1.0);
    img_param.set_global_img_size(m_global_img_size);
    img_param.set_global_img_coord(m_global_img_x_start, m_global_img_y_start);

    auto& numa_util_instance = Naive_NUMA_util::get_instance();
    if (m_comp_cfg.m_mempool_mb > 0) {
        uint64_t mempool_size = m_comp_cfg.m_mempool_mb;
        mempool_size *= std::CPUInfoUtil::I().total_socket_count_;
        mempool_size *= 1024 * 1024;

        printf("socket count: %d\n", std::CPUInfoUtil::I().total_socket_count_);
        printf("mempool_mb: %d\n", m_comp_cfg.m_mempool_mb);
        printf("mempool_size: %lu\n", mempool_size);

        numa_util_instance.create_seq_mem_pool(m_comp_cfg.m_mempool_mb * 1024UL * 1024UL * std::CPUInfoUtil::I().total_socket_count_);
    }

    Partitioner_cscv cscv_partition(img_param, m_angle_group_size, m_img_x_group_size, m_img_y_group_size, m_angle_part, m_img_x_part, m_img_y_part);

    cscv_partition.set_pxg_size(m_pxg_size);
    cscv_partition.set_computation_config(m_comp_cfg);

    cscv_partition.set_block_order(m_block_order);

    Partition_result_cscv partition_result = cscv_partition.get_partition_result();

    Data_holder<Element_type>* data_holder = new Data_holder<Element_type>(img_param, partition_result, &cscv_partition, m_comp_cfg, m_nthreads);

    PRINTF("\n%s\n", get_summary().c_str());

    if (m_comp_cfg.m_validate_generation_only) {
        data_holder->validate_generation();
        return;
    }

    if (!m_comp_cfg.m_disable_validation) {
        data_holder->init_parts_coo();
        for (int i = 0; i < (m_comp_cfg.m_loop_count / 10) + 1; i++) {
            data_holder->compute_parts_coo_y_ax();
        }
    }

    data_holder->free_parts_coo();

    if (m_comp_cfg.m_run_coo_full) {
        data_holder->init_full_coo();
    }

    if (m_comp_cfg.m_run_mkl_csc_full) {
        data_holder->init_full_coo();
        data_holder->init_full_csc();
    }

    if (m_comp_cfg.m_run_mkl_csr_full) {
        data_holder->init_full_coo();
        data_holder->init_full_csr();
    }

    if (!m_comp_cfg.m_run_coo_full) {
        data_holder->free_full_coo();
    }

    if (m_comp_cfg.m_run_cscvb_tea || m_comp_cfg.m_run_cscvb_cat) {
        data_holder->init_blocks_cscvb();
        data_holder->estimate_cscvb_tea_cat_mem_expense();
        data_holder->estimate_block_arr_expense();
    }

    if (m_skip_running)
        return;
    
    data_holder->print_init_timer();

    data_holder->free_system_matrix_gen();

    if (c_vexpand_enabled) {
        PRINTF("Begin y = Ax testing! VEXPAND enabled :)\n");
    } else {
        PRINTF("Begin y = Ax testing! VEXPAND disabled :(\n");
    }

    for (int i = 0; i < m_comp_cfg.m_loop_count; i++) {
        if (m_comp_cfg.m_run_normal) {
            if (m_comp_cfg.m_run_mkl_csc_full)
                data_holder->compute_full_csc_mkl_y_ax();

            if (m_comp_cfg.m_run_mkl_csr_full)
                data_holder->compute_full_csr_mkl_y_ax();

            if (m_comp_cfg.m_run_coo_full)
                data_holder->compute_full_coo_y_ax();

            if (m_comp_cfg.m_run_cscvb_tea)
                data_holder->compute_blocks_cscvb_tea_y_ax(false);

            if (m_comp_cfg.m_run_cscvb_cat)
                data_holder->compute_blocks_cscvb_tea_y_ax(true);
        }

        data_holder->one_loop_finished_y_ax();
    }

    data_holder->compare_y_and_print();
    data_holder->get_calculation_thread_timer_rough();

    data_holder->print_calculation_timer();

    if (m_finish_signal_path.size() != 0) {
        int ret = system(("touch " + m_finish_signal_path).c_str());;
    }

    delete data_holder;
}
