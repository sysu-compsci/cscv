#pragma once

#include "soa.hpp"

template <class Element_type, size_t t_vec_angle, size_t t_px_group, bool t_use_cat>
void tea_blocks_compute_y_ax(Tea_soa_context<Element_type>* context) {
    ASSERT_AND_PRINTF(t_vec_angle == context->m_angle_group_size, "");

    typedef FP_vec<Element_type, t_vec_angle> Vec_angle;
    using Mask_type = typename Vec_angle::Mask_type;

    // const int x_group_size = context->m_x_group_size;
    // const int y_group_size = context->m_y_group_size;
    const int x_group_size = (context->m_x_group_size / 16) * 16;
    const int y_group_size = (context->m_y_group_size / 16) * 16;
    ASSERT_AND_PRINTF(x_group_size == context->m_x_group_size, "");
    ASSERT_AND_PRINTF(y_group_size == context->m_y_group_size, "");

    Element_type y_buffer[context->m_block_max_bin * t_vec_angle] __attribute__((aligned(512)));

    Element_type yt_buffer[context->m_block_max_bin * t_vec_angle] __attribute__((aligned(512)));

    Vec_angle* yt_buffer_vec = (Vec_angle*)&yt_buffer[0];

    Element_type block_x_buffer_2d[y_group_size + 1][x_group_size] __attribute__((aligned(512)));
    const Element_type* block_x_buffer_1d = &block_x_buffer_2d[0][0];

    Element_type* tea_data_evv = &context->m_tea_data->at(0);
    Vec_angle* tea_data_vec_evv = (Vec_angle*)tea_data_evv;
    uint16_t* tea_pxs_evv = &context->m_tea_pxs->at(0);
    uint8_t* tea_bin_offs_evv = &context->m_tea_bin_offs->at(0);
    uint16_t* tea_group_offs_evv = &context->m_tea_group_offs->at(0);

    Mask_type* cat_masks_evv = (Mask_type*)&context->m_cat_masks->at(0);
    Element_type* cat_data_evv = &context->m_cat_data->at(0);
    uint8_t* cat_popcnts_evv = &context->m_cat_popcnts->at(0);

    const Element_type* part_x = &context->m_part_x->m_vec->at(0);
    Element_type* part_y = &context->m_part_y->at(0);
    const int part_x_stride = context->m_part_img_x_size;
    const int part_y_stride = context->m_part_y_stride;
    const int part_img_x_start = context->m_part_start_img_x;
    const int part_img_y_start = context->m_part_start_img_y;
    const int part_angle_start = context->m_part_start_angle;

    const int* block_angle_bin_starts_evv = &context->m_block_angle_bin_starts->at(0);
    const int* block_angle_bin_starts_to_lhs_evv = &context->m_block_angle_bin_starts_to_lhs->at(0);

    // memset(block_x_buffer_2d, 0, sizeof(block_x_buffer_2d));
    for (int i = 0; i < y_group_size + 1; i++) {
        for (int j = 0; j < x_group_size; j++) {
            block_x_buffer_2d[i][j] = 0;
        }
    }

    const int __part_max_bin_per_angle = context->m_part_max_bin_per_angle;
    const int part_max_bin_per_angle = (__part_max_bin_per_angle / t_vec_angle) * t_vec_angle;

    const Element_type* global_x = &context->m_x_input_global->at(0, 0);
    const uint16_t global_x_stride = context->m_x_input_global->m_x;

    // uint64_t* fetch_block_x_cc_ptr;
    // uint64_t* set_yt_to_zero_cc_ptr;
    // uint64_t* do_spmv_y_ax_cc_ptr;
    // uint64_t* transpose_yt_to_y_cc_ptr;
    // uint64_t* reduce_block_y_cc_ptr;
    // uint64_t* temporal_y_process_cc_ptr;
    // uint64_t* spin_cc_ptr;

    if (!t_use_cat) {
        // fetch_block_x_cc_ptr = &(*context->m_tea_y_ax_timers)[(int)Timer_type::GET_BLOCK_X];
        // set_yt_to_zero_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::SET_YT_0)];
        // do_spmv_y_ax_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::DO_SPMV_Y_AX)];
        // transpose_yt_to_y_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::TRANSPOSE_YT)];
        // reduce_block_y_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::REDUCE_BLOCK_Y)];
        // temporal_y_process_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::TEMP_Y_PROCESS)];
        // spin_cc_ptr = &(*context->m_tea_y_ax_timers)[((int)Timer_type::SPIN)];
    } else {
        // fetch_block_x_cc_ptr = &(*context->m_cat_y_ax_timers)[(int)Timer_type::GET_BLOCK_X];
        // set_yt_to_zero_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::SET_YT_0)];
        // do_spmv_y_ax_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::DO_SPMV_Y_AX)];
        // transpose_yt_to_y_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::TRANSPOSE_YT)];
        // reduce_block_y_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::REDUCE_BLOCK_Y)];
        // temporal_y_process_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::TEMP_Y_PROCESS)];
        // spin_cc_ptr = &(*context->m_cat_y_ax_timers)[((int)Timer_type::SPIN)];
    }

    rdtsc_interval();
    Element_type stack_part_y_buffer[part_max_bin_per_angle * context->m_part_angle_count] __attribute__((aligned(512)));
    // Element_type* stack_part_y_buffer = (Element_type*)_mm_malloc(part_max_bin_per_angle * context->m_part_angle_count * sizeof(Element_type), 512);
    for (int i = 0; i < part_max_bin_per_angle * context->m_part_angle_count; i++) {
        stack_part_y_buffer[i] = 0;
    }
    // *temporal_y_process_cc_ptr += rdtsc_interval();

    for (int block_id = 0; block_id < context->m_block_count; block_id++) {
        const uint16_t __block_bin_count = context->m_block_bin_count->at(block_id);
        const uint16_t block_bin_count = (__block_bin_count / t_vec_angle) * t_vec_angle;
        ASSERT_AND_PRINTF(__block_bin_count == block_bin_count, "block bin count cannnot be devided by t_vec_angle(not padded)\n");

        const int block_x_start = context->m_block_x_starts->at(block_id);
        const int block_y_start = context->m_block_y_starts->at(block_id);
        const int block_data_size = context->m_block_data_size->at(block_id);
        const int block_pxg_count = context->m_block_px_group_count->at(block_id);
        const int block_angle_start = context->m_block_angle_starts->at(block_id);

        rdtsc_interval();
        for (int block_img_y = 0; block_img_y < y_group_size; block_img_y++) {
            for (int block_img_x = 0; block_img_x < x_group_size; block_img_x++) {
                block_x_buffer_2d[block_img_y][block_img_x] = part_x[block_img_x + block_x_start - part_img_x_start +
                                                                    (block_img_y + block_y_start - part_img_y_start) * part_x_stride];
            }
        }
        // *fetch_block_x_cc_ptr += rdtsc_interval();

        for (int off = 0; off < block_bin_count; off++)
            yt_buffer_vec[off].set1(0);
        // *set_yt_to_zero_cc_ptr += rdtsc_interval();

        for (int pxg_bin_count = 0; pxg_bin_count < c_pxg_max_bin; pxg_bin_count++) {
            for (int pxg_off = tea_group_offs_evv[pxg_bin_count]; pxg_off < tea_group_offs_evv[pxg_bin_count + 1]; pxg_off++) {
                int pxg_bin_offset = tea_bin_offs_evv[pxg_off];
                int px_offset = pxg_off * t_px_group;

                Vec_angle px_vals[t_px_group];

                for (int px_off0 = 0; px_off0 < t_px_group; px_off0++) {
                    int px_id = tea_pxs_evv[px_offset + px_off0];
                    px_vals[px_off0].set1(block_x_buffer_1d[px_id]);
                }

                for (int bin_off = 0; bin_off < pxg_bin_count; bin_off++) {
                    for (int px_off0 = 0; px_off0 < t_px_group; px_off0++) {
                        int cur_bin_id = pxg_bin_offset + bin_off + px_off0;

                        if constexpr (t_use_cat) {
                            Mask_type mask = *(cat_masks_evv++);
                            uint8_t angle_nnz = __builtin_popcount(mask);

                            Vec_angle tmp_cat_data;

                            if constexpr (t_vec_angle == 16 && std::is_same<float, Element_type>::value && c_vexpand_enabled) {
                                __m512 tmp_cat_data_m = vexpand_loadu_floatv16(mask, cat_data_evv);
                                vstore_floatv16(&tmp_cat_data, tmp_cat_data_m);
                            } else if constexpr (t_vec_angle == 16 && std::is_same<double, Element_type>::value && c_vexpand_enabled) {
                                uint8_t mask_a = mask;
                                uint8_t mask_b = mask >> 8;
                                uint8_t a_nnz = __builtin_popcount(mask_a);
                                __m512d tmp_cat_data_ma = vexpand_loadu_double8(mask_a, cat_data_evv);
                                __m512d tmp_cat_data_mb = vexpand_loadu_double8(mask_b, cat_data_evv + a_nnz);
                                vstore_doublev8(&tmp_cat_data, tmp_cat_data_ma);
                                vstore_doublev8(&tmp_cat_data.m_scalar[8], tmp_cat_data_mb);
                            } else if constexpr (t_vec_angle == 8 && std::is_same<float, Element_type>::value && c_vexpand_enabled) {
                                __m256 tmp_cat_data_m = vexpand_loadu_floatv8(mask, (float*)cat_data_evv);
                                vstore_floatv8((float*)&tmp_cat_data, tmp_cat_data_m);
                            } else if constexpr (t_vec_angle == 8 && std::is_same<double, Element_type>::value && c_vexpand_enabled) {
                                __m512d tmp_cat_data_m = vexpand_loadu_double8(mask, cat_data_evv);
                                vstore_doublev8(&tmp_cat_data, tmp_cat_data_m);
                            } else if constexpr (t_vec_angle == 4 && std::is_same<float, Element_type>::value && c_vexpand_enabled) {
                                __m128 tmp_cat_data_m = vexpand_loadu_floatv4(mask, (float*)cat_data_evv);
                                vstore_floatv4((float*)&tmp_cat_data, tmp_cat_data_m);
                            } else if constexpr (t_vec_angle == 4 && std::is_same<double, Element_type>::value && c_vexpand_enabled) {
                                __m256d tmp_cat_data_m = vexpand_loadu_double4(mask, (double*)cat_data_evv);
                                vstore_doublev4((double*)&tmp_cat_data, tmp_cat_data_m);
                            } else {
                                tmp_cat_data.expand_from_memory(mask, cat_data_evv);
                            }

                            cat_data_evv += angle_nnz;

                            yt_buffer_vec[cur_bin_id].mad_self(px_vals[px_off0], tmp_cat_data);
                        } else {
                            yt_buffer_vec[cur_bin_id].mad_self(px_vals[px_off0], (*(tea_data_vec_evv++)));
                        }
                    }
                }
            }
        }
        // *do_spmv_y_ax_cc_ptr += rdtsc_interval();

        tea_pxs_evv += block_pxg_count * t_px_group;
        tea_bin_offs_evv += block_pxg_count;
        tea_group_offs_evv += c_pxg_max_bin + 1;

        for (int bin = 0; bin < block_bin_count; bin++) {
            for (int angle = 0; angle < t_vec_angle; angle++) {
                y_buffer[bin + angle * block_bin_count] = yt_buffer[bin * t_vec_angle + angle];
            }
        }
        // *transpose_yt_to_y_cc_ptr += rdtsc_interval();

        for (int angle0 = 0; angle0 < t_vec_angle; angle0++) {
            int angle_in_part = block_angle_start + angle0 - part_angle_start;
            int block_angle_offset = block_bin_count * angle0;
            int block_angle_bin_start_to_lhs = block_angle_bin_starts_to_lhs_evv[angle0];

            for (int block_bin_off = 0; block_bin_off < block_bin_count; block_bin_off++) {
                stack_part_y_buffer[part_max_bin_per_angle * angle_in_part + block_bin_off + block_angle_bin_start_to_lhs] += y_buffer[block_angle_offset + block_bin_off];
            }
        }
        // *reduce_block_y_cc_ptr += rdtsc_interval();

        block_angle_bin_starts_evv += t_vec_angle;
        block_angle_bin_starts_to_lhs_evv += t_vec_angle;
    }

    int eff_nthreads = context->m_y_reduction_barrier->get_nthreads();

    // do partition on angle to from
    int nthreads = eff_nthreads, tid = omp_get_thread_num();
    int angle_offs[nthreads + 1];
    for (int i = 0; i < nthreads + 1; i++)
        angle_offs[i] = BLOCK_LOW(i, nthreads, context->m_part_angle_count);

    int global_y_stride = context->m_img_param.m_num_bin;
    Element_type* global_y = &context->m_y_global->at(global_y_stride * context->m_part_start_angle);

    Element_type* my_global_y_start = global_y + angle_offs[tid] * global_y_stride;

    if (context->m_y_reduction_barrier->get_nthreads() == omp_get_num_threads()) {
        #pragma omp barrier
    } else {
        // ring barrier
        context->m_y_reduction_barrier->pass(tid);
        context->m_y_reduction_barrier->wait(tid);
        context->m_y_reduction_barrier->pass(tid);
        context->m_y_reduction_barrier->wait(tid);
    }

    // *spin_cc_ptr += rdtsc_interval();

    for (int i = 0; i < (angle_offs[tid + 1] - angle_offs[tid]) * global_y_stride; i++)
        my_global_y_start[i] = 0;

    for (int i = 0; i < nthreads; i++) {
        int reduction_group_id = (tid + i) % nthreads;

        for (int angle0 = angle_offs[reduction_group_id]; angle0 < angle_offs[reduction_group_id + 1]; angle0++) {
            for (int bin0 = 0; bin0 < part_max_bin_per_angle; bin0++) {
                global_y[global_y_stride * angle0 + bin0 + context->m_part_y_lhss->at(angle0)] +=
                    stack_part_y_buffer[part_max_bin_per_angle * angle0 + bin0];
            }
        }
        // #pragma omp barrier

        context->m_y_reduction_barrier->pass(tid);
        context->m_y_reduction_barrier->wait(tid);
    }

    // _mm_free(stack_part_y_buffer);
    // *temporal_y_process_cc_ptr += rdtsc_interval();
}
