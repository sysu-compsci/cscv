#pragma once

#include "partition.hpp"

#include <omp.h>

#include "arch/pthread_timer.hpp"
#include "cscv/compute.hpp"
#include "cscv/system_matrix_converter.hpp"

template <class Element_type>
Data_holder<Element_type>::Data_holder(const Img_param& img_param, const Partition_result_cscv& partition_result, Partitioner_cscv* partitioner,
                                       const Computation_config& comp_cfg, int nthreads) :
                                       m_img_param(img_param), m_partition_result(partition_result), m_partitioner(partitioner), m_comp_cfg(comp_cfg) {
    set_num_threads(nthreads);

    // build_global_arrs();
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::build_global_arrs), this);

    for (int i = 0; i < partition_result.m_x_part * partition_result.m_y_part * partition_result.m_angle_part; i++) {
        auto* part = m_partitioner->build_part<Element_type>(i);
        if (part != nullptr) {
            part->m_y_tmp = m_y_tmp;
            part->m_x_input = m_x_input;
            m_parts.push_back(part);
        }
    }

    m_part_count = m_parts.size();

    int eff_nthreads = nthreads;
    if (m_part_count < nthreads)
        eff_nthreads = m_part_count;

    m_y_reduction_barriers.push_back(new Ring_fake_barrier(eff_nthreads));

    for (auto* part : m_parts) {
        part->m_y_reduction_barrier = m_y_reduction_barriers[0];
    }

    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::build_thread_local_parts), this);
}

template <class Element_type>
void Data_holder<Element_type>::build_thread_local_parts() {
    #pragma omp for
    for (int part_id = 0; part_id < m_part_count; part_id++) {
        m_partitioner->fill_part(m_parts.at(part_id));
    }
}

template <class Element_type>
void Data_holder<Element_type>::build_global_arrs() {
    if (omp_get_thread_num() != 0)
        return;

    m_x_input = new Image_CT<Element_type>(m_img_param.m_img_size, m_img_param.m_img_size);
    m_x_tmp = new Image_CT<Element_type>(m_img_param.m_img_size, m_img_param.m_img_size);

    m_y_tmp = new Dense_vector<Element_type>(m_img_param.m_num_angle * m_img_param.m_num_bin);
    m_y_std = new Dense_vector<Element_type>(m_img_param.m_num_angle * m_img_param.m_num_bin);

    m_y_tmp->set_zero();
    m_y_std->set_zero();

    for (int i = 0; i < m_img_param.m_img_size; i++) {
        double vy = 0.5 * ((i + 1) * PI - int((i + 1) * PI));
        for (int j = 0; j < m_img_param.m_img_size; j++) {
            double vx = 0.5 * ((j + 1) * PI - int((j + 1) * PI));
            double val = vy + vx;
            m_x_input->at(j, i) = val;
        }
    }

    m_mtx_generator = new System_matrix_generator(m_img_param);
    m_cscv_converter = new System_matrix_converter_cscv<Element_type>(m_img_param, m_partition_result.m_x_group_size,
                                                                      m_partition_result.m_y_group_size, m_partition_result.m_angle_group_size, m_comp_cfg);

    for (auto p : c_result_label_map) {
        m_y_result_table.set_label((int)p.first, p.second);
        m_x_result_table.set_label((int)p.first, p.second);
        m_calc_y_timer_table.set_label((int)p.first, p.second);
        m_calc_x_timer_table.set_label((int)p.first, p.second);
    }

    m_init_process_label_map[Init_process_type::COO_PART] = "COO_PART";
    m_init_process_label_map[Init_process_type::CSR_PART] = "CSR_PART";
    m_init_process_label_map[Init_process_type::CSC_PART] = "CSC_PART";
    m_init_process_label_map[Init_process_type::COO_FULL] = "COO_FULL";
    m_init_process_label_map[Init_process_type::CSR_FULL] = "CSR_FULL";
    m_init_process_label_map[Init_process_type::CSC_FULL] = "CSC_FULL";
    m_init_process_label_map[Init_process_type::COO_BLOCK] = "COO_BLOCK";
    m_init_process_label_map[Init_process_type::CSC_BLOCK] = "CSC_BLOCK";
    m_init_process_label_map[Init_process_type::CSR_BLOCK] = "CSR_BLOCK";
    m_init_process_label_map[Init_process_type::CSCVB_BLOCK] = "CSCVB_BLOCK";

    for (auto p : m_init_process_label_map) {
        m_init_timer_table.set_label((int)p.first, p.second);
    }
}

template <class Element_type>
void Data_holder<Element_type>::reduce_y_tmp_from_parts() {
    uint64_t* part_reduction_time_ptr = nullptr;
    if (m_current_result_type == Result_type::CSCVB_TEA_BLOCK || m_current_result_type == Result_type::CSCVB_CAT_BLOCK) {
        part_reduction_time_ptr = &(Threads_timer_map::get_local()->get_time_ref((int)SPMV_direction::Y_AX, (int)m_current_result_type, (int)Timer_type::REDUCE_PART_Y));
    }

    rdtsc_interval();

    m_y_tmp->set_zero_omp_for();
    #pragma omp barrier

    for (int part_id = 0; part_id < m_parts.size(); part_id++) {
        Part<Element_type>* part = m_parts[part_id];
        if (part->is_empty())
            continue;

        int y_reduction_dst = part->m_start_angle * m_img_param.m_num_bin;
        Element_type *dst = &m_y_tmp->at(y_reduction_dst);
        const Element_type *src = &part->m_part_detector->at(0);

        #pragma omp for nowait
        for (int angle0 = 0; angle0 < part->m_angle_count; angle0++) {
            for (int bin = part->m_y_lhss->at(angle0); bin <= part->m_y_rhss->at(angle0); bin++) {
                dst[angle0 * m_img_param.m_num_bin + bin] += src[angle0 * m_img_param.m_num_bin + bin];
            }
        }

        #pragma omp barrier
    }

    if (part_reduction_time_ptr != nullptr)
        *part_reduction_time_ptr += rdtsc_interval();
}

template <class Element_type>
Data_holder<Element_type>::~Data_holder() {
    if (m_mtx_generator)
        delete m_mtx_generator;
    if (m_cscv_converter)
        delete m_cscv_converter;

    if (m_coo_full)
        delete m_coo_full;
    if (m_csc_full)
        delete m_csc_full;
    if (m_csr_full)
        delete m_csr_full;

    if (m_csr_mkl_full) {
        mkl_sparse_destroy(*m_csr_mkl_full);
        delete m_csr_mkl_full;
    }
    if (m_csc_mkl_full) {
        mkl_sparse_destroy(*m_csc_mkl_full);
        delete m_csc_mkl_full;
    }
    if (m_csr_trans_mkl_full) {
        mkl_sparse_destroy(*m_csr_trans_mkl_full);
        delete m_csr_trans_mkl_full;
    }
    if (m_csc_trans_mkl_full) {
        mkl_sparse_destroy(*m_csc_trans_mkl_full);
        delete m_csc_trans_mkl_full;
    }

    if (m_x_input)
        delete m_x_input;
    if (m_x_tmp)
        delete m_x_tmp;
    if (m_y_tmp)
        delete m_y_tmp;
    if (m_y_std)
        delete m_y_std;

    for (Part<Element_type>* part : m_parts)
        delete part;

    if (m_thread_pool)
        delete m_thread_pool;
}

template <class Element_type>
void Data_holder<Element_type>::thread_local_init() {
    Threads_timer_map::register_thread(omp_get_thread_num());
}

template <class Element_type>
void Data_holder<Element_type>::set_num_threads(int num_threads) {
    m_num_threads = num_threads;
    mkl_set_dynamic(0);
    mkl_set_num_threads(1);
    omp_set_nested(0);

    if (m_thread_pool)
        delete m_thread_pool;
    m_thread_pool = new OMP_thread_pool(num_threads, m_comp_cfg.m_stack_size_mb);

    Threads_timer_map::init_in_sequential_area();

    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&Data_holder<Element_type>::thread_local_init), this);

    PRINTF("num_threads set to %d\n", num_threads);

    m_thread_calculation_timers.clear();
    // m_thread_calculation_timers.resize(num_threads);
    m_thread_reverse_calculation_timers.clear();
    // m_thread_reverse_calculation_timers.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        m_thread_calculation_timers.emplace_back();
        m_thread_reverse_calculation_timers.emplace_back();
    }

    // ASSERT_AND_PRINTF(num_threads == mkl_get_max_threads(), "thread count %d != %d\n", num_threads, mkl_get_max_threads());
}

template <class Element_type>
void Data_holder<Element_type>::one_loop_finished_y_ax() {
    m_timers_maps_y_ax.push_back(dump_timers_map_and_clear());
}

template <class Element_type>
std::string Data_holder<Element_type>::get_workload_filename() const {
    // thread, img_size, num_angle, num_bin, delta_angle, block x, block y, block a, precision, pxg_size, loop

    std::string ret;

    ret += strprintf("%d-t__", m_num_threads);
    ret += strprintf("%d-bx__", m_partition_result.m_x_group_size);
    ret += strprintf("%d-by__", m_partition_result.m_y_group_size);
    ret += strprintf("%d-ba__", m_partition_result.m_angle_group_size);
    ret += strprintf("%d-pxg__", m_partition_result.m_pxg_size);
    if (std::is_same<Element_type, double>::value) {
        ret += "d-p__";
    } else if (std::is_same<Element_type, float>::value) {
        ret += "f-p__";
    }
    ret += strprintf("%d-l__", m_comp_cfg.m_loop_count);

    ret += strprintf("%d-img_size__", m_img_param.m_img_size);
    ret += strprintf("%d-num_angle__", m_img_param.m_num_angle);
    ret += strprintf("%d-num_bin__", m_img_param.m_num_bin);
    ret += strprintf("%d-global_img_size__", m_img_param.m_global_img_size);
    ret += strprintf("%d-global_img_x_start__", m_img_param.m_global_img_x_start);
    ret += strprintf("%d-global_img_y_start__", m_img_param.m_global_img_y_start);

    ret += strprintf("%f-delta_angle__", m_img_param.m_delta_angle);

    ret += strprintf("%d-xp__", m_partition_result.m_x_part);
    ret += strprintf("%d-yp__", m_partition_result.m_y_part);
    ret += strprintf("%d-ap__", m_partition_result.m_angle_part);

    ret += ".log";

    return ret;
}

template <class Element_type>
void Data_holder<Element_type>::print_calculation_timer() {
    printf("%s\n", m_calc_y_timer_table.get_summary_statistics_string("timer y = Ax").c_str());

    Logger::get_instance().set_filename(get_workload_filename());

    Logger::get_instance().write(strprintf("[Performance Timer]\n"));

    std::vector<int> y_ax_cols = m_calc_y_timer_table.get_effective_keys();
    for (auto col_id : y_ax_cols) {
        if (col_id == (int)Result_type::PART_X_REDUCTION || col_id == (int)Result_type::PART_Y_REDUCTION)
            continue;
        std::string label = c_result_label_map.at((Result_type)col_id);
        double min_time = m_calc_y_timer_table.get_column(col_id).m_results.get_min();
        double median_time = m_calc_y_timer_table.get_column(col_id).m_results.get_median();
        double average_time = m_calc_y_timer_table.get_column(col_id).m_results.get_average();
        Logger::get_instance().write(strprintf("%s yax\tmin\t%f\n", label.c_str(), min_time));
        Logger::get_instance().write(strprintf("%s yax\tmed\t%f\n", label.c_str(), median_time));
        Logger::get_instance().write(strprintf("%s yax\tavg\t%f\n", label.c_str(), average_time));
    }

    std::stringstream imba_ss;

    auto imba_keys = {Result_type::CSCVB_CAT_BLOCK, Result_type::CSCVB_TEA_BLOCK};
    auto imba_directions = {std::make_tuple(SPMV_direction::Y_AX, &m_y_result_table, &m_calc_y_timer_table, &m_timers_maps_y_ax)};

    for (auto direction_tuple : imba_directions) {
        for (auto key : imba_keys) {
            if (std::get<1>(direction_tuple)->check_if_key_exist((int)key)) {
                int min_pos = std::get<2>(direction_tuple)->get_column((int)key).m_results.get_min_position();
                const Timers_instance_map& ref = std::get<3>(direction_tuple)->at(min_pos);

                imba_ss << get_timer_map_str_for_particular_key(ref, std::get<0>(direction_tuple), key);
            }
        }
    }

    Logger::get_instance().write(strprintf("[Imba Timer]\n"));

    if (m_comp_cfg.m_print_thread_imba)
        printf("%s\n", imba_ss.str().c_str());
    Logger::get_instance().write(imba_ss.str());

    if (m_comp_cfg.m_run_normal) {
        Logger::get_instance().write(strprintf("[Tea Cat Performance]\n"), true);
        if (m_comp_cfg.m_run_cscvb_tea) {
            double min_time = m_calc_y_timer_table.get_column((int)Result_type::CSCVB_TEA_BLOCK).m_results.get_min();
            double gflops = (m_cscv_converter->get_original_nnz() * 2.0 / min_time) / (1000.0 * 1000.0 * 1000.0);
            double eff_mbw = (m_tea_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0)) / min_time;

            Logger::get_instance().write(strprintf("cscvb_tea\tgflops\t%f\n", gflops), true);
            Logger::get_instance().write(strprintf("cscvb_tea\teff_mbw\t%f\n", eff_mbw), true);
        }

        if (m_comp_cfg.m_run_cscvb_cat) {
            double min_time = m_calc_y_timer_table.get_column((int)Result_type::CSCVB_CAT_BLOCK).m_results.get_min();
            double gflops = (m_cscv_converter->get_original_nnz() * 2.0 / min_time) / (1000.0 * 1000.0 * 1000.0);
            double eff_mbw = (m_cat_iter_bytes_expense / (1024.0 * 1024.0 * 1024.0)) / min_time;

            Logger::get_instance().write(strprintf("cscvb_cat\tgflops\t%f\n", gflops), true);
            Logger::get_instance().write(strprintf("cscvb_cat\teff_mbw\t%f\n", eff_mbw), true);
        }

        if (m_comp_cfg.m_run_mkl_csr_full) {
            double min_time = m_calc_y_timer_table.get_column((int)Result_type::MKL_CSR_FULL).m_results.get_min();
            double gflops = (m_original_nnz * 2.0 / min_time) / (1000.0 * 1000.0 * 1000.0);
            double eff_mbw = ((m_csr_full->get_bytes() + m_x_tmp->m_vec->get_bytes() + m_y_tmp->get_bytes()) / (1024.0 * 1024.0 * 1024.0)) / min_time;

            Logger::get_instance().write(strprintf("mkl_csr\tgflops\t%f\n", gflops), true);
            Logger::get_instance().write(strprintf("mkl_csr\teff_mbw\t%f\n", eff_mbw), true);
        }

        if (m_comp_cfg.m_run_mkl_csc_full) {
            double min_time = m_calc_y_timer_table.get_column((int)Result_type::MKL_CSC_FULL).m_results.get_min();
            double gflops = (m_original_nnz * 2.0 / min_time) / (1000.0 * 1000.0 * 1000.0);
            double eff_mbw = ((m_csc_full->get_bytes() + m_x_tmp->m_vec->get_bytes() + m_y_tmp->get_bytes()) / (1024.0 * 1024.0 * 1024.0)) / min_time;

            Logger::get_instance().write(strprintf("mkl_csc\tgflops\t%f\n", gflops), true);
            Logger::get_instance().write(strprintf("mkl_csc\teff_mbw\t%f\n", eff_mbw), true);
        }
    }

    Logger::get_instance().close_file();
}

template <class Element_type>
void Data_holder<Element_type>::print_init_timer() {
    printf("%s\n", m_init_timer_table.get_summary_statistics_string("timer init").c_str());
}

template <class Element_type>
void Data_holder<Element_type>::compare_y_and_print() {
    printf("%s\n", m_y_result_table.get_summary_diff_string("y diff").c_str());
}

template <class Element_type>
void Data_holder<Element_type>::compare_x_and_print() {
    printf("%s\n", m_x_result_table.get_summary_diff_string("x diff").c_str());
}

template <class Element_type>
void Data_holder<Element_type>::get_calculation_thread_timer_rough() {
    // check thread 0's record for detail
    const auto& t0_info = m_thread_calculation_timers[0];

    String_table st("calc thread accumulation info");

    for (auto p : t0_info) {
        double max_spin_time = 0;
        double min_spin_time = 99999999;
        double max_time = 0;
        double spin_sum = 0;

        for (int i = 0; i < m_num_threads; i++) {
            ASSERT_AND_PRINTF(m_thread_calculation_timers[i].count(p.first) != 0, "in thread %d, %s not found!\n", i, c_result_label_map.at(p.first).c_str());
            double spin_time = m_thread_calculation_timers[i].at(p.first).m_spin_time;
            double computation_time = m_thread_calculation_timers[i].at(p.first).m_calculation_time;

            max_spin_time = std::max(max_spin_time, spin_time);
            max_time = std::max(max_time, spin_time + computation_time);

            spin_sum += spin_time;
        }

        st.set_value(c_result_label_map.at(p.first), "spin ratio", spin_sum / (max_time * m_num_threads));
        st.set_value(c_result_label_map.at(p.first), "spin min ratio", min_spin_time / max_time);
        st.set_value(c_result_label_map.at(p.first), "spin max ratio", max_spin_time / max_time);
    }

    printf("%s\n", st.to_string().c_str());
}

template <class Element_type>
int Data_holder<Element_type>::get_max_cscv_block_bin_count() const {
    int ret = 0;
    for (auto part : m_parts) {
        for (auto block : part->m_blocks) {
            ret = std::max(ret, block->m_cscvb_block->m_block_angle_bin_count);
        }
    }
    return ret;
}

template <class Element_type>
void Data_holder<Element_type>::estimate_block_arr_expense() const {
    int yt_size_sum = 0, px_vals_size_sum = 0, x_tmp_size_sum = 0;

    for (Part<Element_type>* part : m_parts) {
        yt_size_sum += part->m_tea_soa_context->m_block_max_bin * part->m_block_angle * sizeof(Element_type);
        px_vals_size_sum += part->m_block_angle * sizeof(Element_type);
        x_tmp_size_sum += part->m_block_x * part->m_block_y * sizeof(Element_type);
    }

    printf("[SOA] TMP arr size: yt: %d, px_vals: %d, x_tmp: %d, with config %s\n",
            yt_size_sum, px_vals_size_sum, x_tmp_size_sum, get_workload_filename().c_str());
}

template <class Element_type>
void Data_holder<Element_type>::free_system_matrix_gen() {
    m_thread_pool->run_by_all_blocked(force_cast_pointer<OMP_thread_pool::Op_func>(&free_system_matrix_gen_tmp_data), nullptr);
}

template <class Element_type>
void Data_holder<Element_type>::validate_generation() {
    // generate blocks coo, parts coo and full coo
    init_full_coo();
    init_parts_coo();
    init_blocks_coo_debug_only();

    std::map<std::pair<int, int>, std::pair<int, int> > mark_table;

    for (int i = 0; i < m_coo_full->m_nz_count; i++) {
        int row = m_coo_full->m_nz_row_idx[i];
        int col = m_coo_full->m_nz_col_idx[i];
        mark_table[{row, col}] = {1, 1};
    }

    for (Part<Element_type>* part : m_parts) {
        for (int i = 0; i < part->m_coo_part->m_nz_count; i++) {
            int row = part->m_coo_part->m_nz_row_idx[i];
            int col = part->m_coo_part->m_nz_col_idx[i];
            int global_x = col % part->m_size_image_x + part->m_start_x;
            int global_y = col / part->m_size_image_x + part->m_start_y;
            int global_col = global_x + global_y * m_img_param.m_img_size;
            int global_row = part->m_start_angle * part->m_num_bin + row;

            if (mark_table.count({global_row, global_col}) == 0) {
                printf("Not found, in part, global row = %d, global col = %d\n", global_row, global_row);
            } else {
                mark_table.at({global_row, global_col}).first -= 1;
            }
        }

        for (Block<Element_type> * block : part->m_blocks) {
            for (int i = 0; i < block->m_coo_block->m_nz_count; i++) {
                int row = block->m_coo_block->m_nz_row_idx[i];
                int col = block->m_coo_block->m_nz_col_idx[i];
                int global_x = col % block->m_size_image_x + block->m_start_x;
                int global_y = col / block->m_size_image_x + block->m_start_y;
                int global_col = global_x + global_y * m_img_param.m_img_size;
                int global_row = block->m_start_angle * part->m_num_bin + row;

                if (mark_table.count({global_row, global_col}) == 0) {
                    printf("Not found, in block, global row = %d, global col = %d\n", global_row, global_row);
                } else {
                    mark_table.at({global_row, global_col}).second -= 1;
                }
            }
        }
    }

    int part_diff_sum = 0, block_diff_sum = 0;

    for (auto kv : mark_table) {
        part_diff_sum += std::abs(kv.second.first);
        block_diff_sum += std::abs(kv.second.second);
    }

    int parts_nnz = 0, blocks_nnz = 0;
    for (Part<Element_type>* part : m_parts) {
        parts_nnz += part->m_coo_part->m_nz_count;
        for (Block<Element_type> * block : part->m_blocks) {
            blocks_nnz += block->m_coo_block->m_nz_count;
        }
    }
    printf("full nnz = %d, part nnz = %d, block nnz = %d\n", m_coo_full->m_nz_count, parts_nnz, blocks_nnz);

    printf("part diff sum = %d, block diff sum = %d\n", part_diff_sum, block_diff_sum);
}

template <class Element_type>
Part<Element_type>* Partitioner_cscv::build_part(int part_id) {
    // get the img_x, img_y and angle part
    int angle_part_id = part_id / (m_x_part * m_y_part);
    int x_part_id = part_id % m_x_part;
    int y_part_id = (part_id / m_x_part) % m_y_part;

    int m_size_image_x = (m_img_x_block_id_offsets[x_part_id + 1] - m_img_x_block_id_offsets[x_part_id]) * m_x_group_size;
    int m_size_image_y = (m_img_y_block_id_offsets[y_part_id + 1] - m_img_y_block_id_offsets[y_part_id]) * m_y_group_size;
    int m_angle_count = (m_angle_block_id_offsets[angle_part_id + 1] - m_angle_block_id_offsets[angle_part_id]) * m_angle_group_size;
    if (m_size_image_x == 0 || m_size_image_y == 0 || m_angle_count == 0)
        return nullptr;

    Part<Element_type>* part = new Part<Element_type>;

    // part->m_img_part_id = x_part_id + y_part_id * m_x_part;

    // setup the part info
    part->m_start_x = m_img_x_block_id_offsets[x_part_id] * m_x_group_size;
    part->m_start_y = m_img_y_block_id_offsets[y_part_id] * m_y_group_size;
    part->m_start_angle = m_angle_block_id_offsets[angle_part_id] * m_angle_group_size;
    part->m_x_group_size = m_x_group_size;
    part->m_y_group_size = m_y_group_size;
    part->m_angle_group_size = m_angle_group_size;
    part->m_pxg_size = m_pxg_size;

    part->m_size_image_x = (m_img_x_block_id_offsets[x_part_id + 1] - m_img_x_block_id_offsets[x_part_id]) * m_x_group_size;
    part->m_size_image_y = (m_img_y_block_id_offsets[y_part_id + 1] - m_img_y_block_id_offsets[y_part_id]) * m_y_group_size;
    part->m_angle_count = (m_angle_block_id_offsets[angle_part_id + 1] - m_angle_block_id_offsets[angle_part_id]) * m_angle_group_size;

    part->m_num_bin = m_img_param.m_num_bin;

    // create blocks
    part->m_block_x = m_img_x_block_id_offsets[x_part_id + 1] - m_img_x_block_id_offsets[x_part_id];
    part->m_block_y = m_img_y_block_id_offsets[y_part_id + 1] - m_img_y_block_id_offsets[y_part_id];
    part->m_block_angle = m_angle_block_id_offsets[angle_part_id + 1] - m_angle_block_id_offsets[angle_part_id];

    part->m_comp_cfg = m_comp_cfg;

    part->m_img_param = m_img_param;

    return part;
}

template <class Element_type>
void Partitioner_cscv::fill_part(Part<Element_type>* part) {
    part->m_part_detector = new Dense_vector<Element_type>(part->m_angle_count * part->m_num_bin);
    assert(part->m_part_detector->get_size() != 0);

    part->m_y_lhss = new Dense_vector<int>(part->m_angle_count);
    part->m_y_rhss = new Dense_vector<int>(part->m_angle_count);

    for (int i = 0; i < part->m_angle_count; i++) {
        part->m_y_lhss->at(i) = part->m_num_bin;
        part->m_y_rhss->at(i) = 0;
    }

    if (part->m_angle_count + part->m_start_angle > m_img_param.m_num_angle) {
        part->m_angle_count = m_img_param.m_num_angle - part->m_start_angle;
    }

    // attach part local memory
    part->m_part_image = new Image_CT<Element_type>(part->m_size_image_x, part->m_size_image_y);
    part->m_part_detector->set_zero();


    // should these be member variables
    int angle_block_stride, x_block_stride, y_block_stride;

    if (m_block_order == Block_order::ANGLE_X_Y) {
        y_block_stride = 1;
        x_block_stride = part->m_block_y;
        angle_block_stride = part->m_block_x * part->m_block_y;
    } else if (m_block_order == Block_order::ANGLE_Y_X) {
        x_block_stride = 1;
        y_block_stride = part->m_block_x;
        angle_block_stride = part->m_block_x * part->m_block_y;
    } else if (m_block_order == Block_order::X_ANGLE_Y) {
        y_block_stride = 1;
        angle_block_stride = part->m_block_y;
        x_block_stride = part->m_block_y * part->m_block_angle;
    } else if (m_block_order == Block_order::X_Y_ANGLE) {
        angle_block_stride = 1;
        y_block_stride = part->m_block_angle;
        x_block_stride = part->m_block_angle * part->m_block_y;
    } else if (m_block_order == Block_order::Y_ANGLE_X) {
        x_block_stride = 1;
        angle_block_stride = part->m_block_x;
        y_block_stride = part->m_block_x * part->m_block_angle;
    } else if (m_block_order == Block_order::Y_X_ANGLE) {
        angle_block_stride = 1;
        x_block_stride = part->m_block_angle;
        y_block_stride = part->m_block_x * part->m_block_angle;
    } else {
        assert(false);
    }

    for (int block_id = 0; block_id < part->m_block_x * part->m_block_y * part->m_block_angle; block_id++) {
        Block<Element_type>* block = new Block<Element_type>;

        part->m_blocks.push_back(block);
    }

    for (int angle_block_id = 0; angle_block_id < part->m_block_angle; angle_block_id++) {
        // check if 
        int effective_angle_count = m_angle_group_size;
        if (part->m_start_angle + (angle_block_id + 1) * m_angle_group_size > m_img_param.m_num_angle) {
            effective_angle_count = m_img_param.m_num_angle - (part->m_start_angle + angle_block_id * m_angle_group_size);
            ASSERT_AND_PRINTF(effective_angle_count < m_angle_group_size && effective_angle_count > 0, "wrong effective angle count %d!\n", effective_angle_count);
        }

        for (int x_block_id = 0; x_block_id < part->m_block_x; x_block_id++) {
            for (int y_block_id = 0; y_block_id < part->m_block_y; y_block_id++) {
                int block_id = x_block_id * x_block_stride + y_block_id * y_block_stride + angle_block_id * angle_block_stride;

                Block<Element_type>* block = part->m_blocks.at(block_id);

                // setup the block info
                block->m_size_image_x = m_x_group_size;
                block->m_size_image_y = m_y_group_size;
                block->m_angle_count = effective_angle_count;

                block->m_start_x = part->m_start_x + x_block_id * m_x_group_size;
                block->m_start_y = part->m_start_y + y_block_id * m_y_group_size;
                block->m_start_angle = part->m_start_angle + angle_block_id * m_angle_group_size;
                block->m_angle_group_size = m_angle_group_size;
                block->m_pxg_size = m_pxg_size;
            }
        }
    }
}

// template <class Element_type>
// std::vector<Part<Element_type>*> Partitioner_cscv::build_parts() {
//     std::vector<Part<Element_type>*> parts;

//     for (int i = 0; i < get_logical_part_count(); i++) {
//         auto* part = build_part<Element_type>(i);
//         if (part != nullptr)
//             parts.push_back(build_part<Element_type>(i));
//     }

//     return parts;
// }
