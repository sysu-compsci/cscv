#include "system_matrix_converter.hpp"

#include <algorithm>

#if defined(__x86_64__) || defined(__i386__)
#include <nmmintrin.h>
#endif

#include "arch/naive_numa_util.hpp"
#include "cscv/flags.hpp"

template <class Element_type>
CSCV_block_mem_pool<Element_type>::CSCV_block_mem_pool(const std::vector<CSCV_block_statistics*>& stas) {

    // 只计算 size，first use 的时候才创建
    for (auto* sta : stas) {
        m_tea_data_size += sta->m_nnz_tea;
        m_tea_bin_offs_size += sta->m_pxgs->get_bin_offs().size();
        m_tea_pxs_size += sta->m_pxgs->get_pixels().size();
        m_cat_data_size += sta->m_nnz_native;
        m_cat_masks_size += sta->m_masks_bytes;
        m_cat_popcnts_size += sta->m_popcnts_size;
    }
}

template <class Element_type>
Dense_vector<Element_type>* CSCV_block_mem_pool<Element_type>::get_next_tea_data(size_t size) {
    if (m_tea_data_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_tea_data_pool = new Dense_vector<Element_type>(m_tea_data_size, (Element_type*)Naive_NUMA_util::get_instance().allocate_with_numa_locality(m_tea_data_size * sizeof(Element_type)));
        } else {
            m_tea_data_pool = new Dense_vector<Element_type>(m_tea_data_size);
        }
        m_tea_data_pool->set_zero();
    }

    Element_type* ptr = &m_tea_data_pool->at(m_tea_data_allocated);
    m_tea_data_allocated += size;
    ASSERT_AND_PRINTF(m_tea_data_allocated <= m_tea_data_size, "insufficient data size\n");

    return new Dense_vector<Element_type>(size, ptr);
}

template <class Element_type>
Dense_vector<uint16_t>* CSCV_block_mem_pool<Element_type>::get_next_tea_pxs(size_t size) {
    if (m_tea_pxs_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_tea_pxs_pool = new Dense_vector<uint16_t>(m_tea_pxs_size, (uint16_t*)Naive_NUMA_util::get_instance().allocate_with_numa_locality(m_tea_pxs_size * sizeof(uint16_t)));
        } else {
            m_tea_pxs_pool = new Dense_vector<uint16_t>(m_tea_pxs_size);
        }
        m_tea_pxs_pool->set_zero();
    }

    uint16_t* ptr = &m_tea_pxs_pool->at(m_tea_pxs_allocated);
    m_tea_pxs_allocated += size;
    ASSERT_AND_PRINTF(m_tea_pxs_allocated <= m_tea_pxs_size, "insufficient data size\n");

    return new Dense_vector<uint16_t>(size, ptr);
}

template <class Element_type>
Dense_vector<uint8_t>* CSCV_block_mem_pool<Element_type>::get_next_tea_bin_offs(size_t size) {
    if (m_tea_bin_offs_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_tea_bin_offs_pool = new Dense_vector<uint8_t>(m_tea_bin_offs_size, (uint8_t*)Naive_NUMA_util::get_instance().allocate_with_numa_locality(m_tea_bin_offs_size * sizeof(uint8_t)));
        } else {
            m_tea_bin_offs_pool = new Dense_vector<uint8_t>(m_tea_bin_offs_size);
        }
        m_tea_bin_offs_pool->set_zero();
    }

    uint8_t* ptr = &m_tea_bin_offs_pool->at(m_tea_bin_offs_allocated);
    m_tea_bin_offs_allocated += size;
    ASSERT_AND_PRINTF(m_tea_bin_offs_allocated <= m_tea_bin_offs_size, "insufficient data size\n");

    return new Dense_vector<uint8_t>(size, ptr);
}

template <class Element_type>
Dense_vector<Element_type>* CSCV_block_mem_pool<Element_type>::get_next_cat_data(size_t size) {
    if (m_cat_data_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_cat_data_pool = new Dense_vector<Element_type>(m_cat_data_size + 64, (Element_type*)Naive_NUMA_util::get_instance().allocate_with_numa_locality((m_cat_data_size + 64) * sizeof(Element_type)));
        } else {
            m_cat_data_pool = new Dense_vector<Element_type>(m_cat_data_size + 64);
        }
        m_cat_data_pool->set_zero();
    }

    Element_type* ptr = &m_cat_data_pool->at(m_cat_data_allocated);
    m_cat_data_allocated += size;
    ASSERT_AND_PRINTF(m_cat_data_allocated <= m_cat_data_size, "insufficient data size\n");

    return new Dense_vector<Element_type>(size, ptr);
}

template <class Element_type>
Dense_vector<std::byte>* CSCV_block_mem_pool<Element_type>::get_next_cat_masks(size_t size) {
    if (m_cat_masks_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_cat_masks_pool = new Dense_vector<std::byte>(m_cat_masks_size, (std::byte*)Naive_NUMA_util::get_instance().allocate_with_numa_locality(m_cat_masks_size * sizeof(std::byte)));
        } else {
            m_cat_masks_pool = new Dense_vector<std::byte>(m_cat_masks_size);
        }
        m_cat_masks_pool->set_zero();
    }

    std::byte* ptr = &m_cat_masks_pool->at(m_cat_masks_allocated);
    m_cat_masks_allocated += size;
    ASSERT_AND_PRINTF(m_cat_masks_allocated <= m_cat_masks_size, "insufficient data size\n");

    return new Dense_vector<std::byte>(size, ptr);
}

template <class Element_type>
Dense_vector<uint8_t>* CSCV_block_mem_pool<Element_type>::get_next_cat_popcnts(size_t size) {
    if (m_cat_popcnts_pool == nullptr) {
        if (Naive_NUMA_util::get_instance().seq_mem_pool_exist()) {
            m_cat_popcnts_pool = new Dense_vector<uint8_t>(m_cat_popcnts_size, (uint8_t*)Naive_NUMA_util::get_instance().allocate_with_numa_locality(m_cat_popcnts_size * sizeof(uint8_t)));
        } else {
            m_cat_popcnts_pool = new Dense_vector<uint8_t>(m_cat_popcnts_size);
        }
        m_cat_popcnts_pool->set_zero();
    }

    uint8_t* ptr = &m_cat_popcnts_pool->at(m_cat_popcnts_allocated);
    m_cat_popcnts_allocated += size;
    ASSERT_AND_PRINTF(m_cat_popcnts_allocated <= m_cat_popcnts_size, "insufficient data size\n");

    return new Dense_vector<uint8_t>(size, ptr);
}

template <class Element_type>
void System_matrix_converter_cscv<Element_type>::clear_statictics() {
    m_nnz = 0;
    m_cscv_element_count = 0;
}

template <class Element_type>
void System_matrix_converter_cscv<Element_type>::get_pixel_bin_ranges(const CSC_matrix<Element_type>* csc, int pixel_id, int* pixel_angle_bin_lhs, int* pixel_angle_bin_rhs) {
    int col = pixel_id;
    ASSERT_AND_PRINTF(csc->m_col_offsets[col + 1] - csc->m_col_offsets[col] > 0,
                        "a pixel should have nnz bins in a specific angle\n");

    int first_row, last_row;

    int first_bin;
    int last_angle_id = -1, last_bin_id;

    for (int offset = csc->m_col_offsets[col]; offset < csc->m_col_offsets[col + 1]; offset++) {
        int row = csc->m_row_idxs[offset];
        int angle_id_in_block = row / m_img_param.m_num_bin;
        ASSERT_AND_PRINTF(angle_id_in_block >= 0, "");
        int bin_id = row % m_img_param.m_num_bin;

        bool first_bin_in_angle = false;
        if (offset == csc->m_col_offsets[col] || angle_id_in_block != last_angle_id)
            first_bin_in_angle = true;

        if (!first_bin_in_angle) {
            ASSERT_AND_PRINTF(bin_id == last_bin_id + 1, "bin id should be continual\n");
        } else {
            ASSERT_AND_PRINTF(angle_id_in_block == last_angle_id + 1, "angle id should be continual, %d != %d + 1\n", angle_id_in_block, last_angle_id);
            if (last_angle_id >= 0)
                pixel_angle_bin_rhs[last_angle_id] = last_bin_id;
            pixel_angle_bin_lhs[angle_id_in_block] = bin_id;
        }

        last_angle_id = angle_id_in_block;
        last_bin_id = bin_id;

        ASSERT_AND_PRINTF(last_angle_id != -1, "");
    }
    pixel_angle_bin_rhs[last_angle_id] = last_bin_id;
}

template <class Element_type>
int System_matrix_converter_cscv<Element_type>::pad_lhs_and_get_bin_shift(const int* ref_lhs, int* lhs_to_pad, int block_angle_count) {
    int tmp_diff[block_angle_count];

    for (int i = 0; i < block_angle_count; i++) {
        tmp_diff[i] = ref_lhs[i] - lhs_to_pad[i];
    }

    // int diff_min = get_arr_min(tmp_diff, block_angle_count);
    int diff_max = get_arr_max(tmp_diff, block_angle_count);
    // int left_shift_max = diff_max - diff_min;

    for (int i = 0; i < block_angle_count; i++) {
        lhs_to_pad[i] -= diff_max - tmp_diff[i];
        ASSERT_AND_PRINTF(lhs_to_pad[0] - ref_lhs[0] == lhs_to_pad[i] - ref_lhs[i], "shift mismatch!\n");
    }

    return lhs_to_pad[0] - ref_lhs[0];
}

static inline int get_effective_bin_count(const std::vector<std::vector<PX_bin_count_pair> >& bin_slots) {
    std::set<int> px_set;

    for (auto bin_slot : bin_slots) {
        for (auto p : bin_slot) {
            ASSERT_AND_PRINTF(px_set.count(p.m_px) == 0, "duplicate px %d\n", p.m_px);
            px_set.insert(p.m_px);
        }
    }

    return px_set.size();
}

static inline PX_grouping* get_grouping_blocked(const std::vector<std::vector<PX_bin_count_pair> >& bin_slots, int px_group) {
    PX_grouping* ret = new PX_grouping(px_group);

    int bin_block_count = div_and_ceil(bin_slots.size(), px_group);

    for (int block = 0; block < bin_block_count; block++) {
        // get max count
        int block_bin_start = block * px_group;
        int max_px_count = 0;
        for (int bin = block_bin_start; bin < block_bin_start + px_group; bin++) {
            if (bin < bin_slots.size()) {
                max_px_count = std::max(max_px_count, (int)bin_slots.at(bin).size());
            }
        }

        // build group
        for (int px_off = 0; px_off < max_px_count; px_off++) {
            std::vector<int> pixels;
            int bin_count = 0;

            for (int bin = block_bin_start; bin < block_bin_start + px_group; bin++) {
                if (bin < bin_slots.size() && px_off < bin_slots.at(bin).size()) {
                    bin_count = std::max(bin_count, bin_slots.at(bin).at(px_off).m_bin_count);
                    pixels.push_back(bin_slots.at(bin).at(px_off).m_px);
                } else {
                    pixels.push_back(PX_grouping::EMPTY_PX);
                }
            }

            ret->append_group(block_bin_start, bin_count, pixels);
        }
    }

    return ret;
}

static inline PX_grouping* get_grouping_greedy_naive(const std::vector<std::vector<PX_bin_count_pair> >& bin_slots, int px_group) {
    PX_grouping* ret = new PX_grouping(px_group);

    std::vector<int> occupied_px_counts;
    for (int bin = 0; bin < bin_slots.size(); bin++) {
        occupied_px_counts.push_back(0);
    }

    for (int bin_outer = 0; bin_outer < bin_slots.size(); bin_outer++) {
        while (occupied_px_counts.at(bin_outer) < bin_slots.at(bin_outer).size()) {
            std::vector<int> pixels;
            int bin_count = 0;

            for (int bin = bin_outer; bin < bin_outer + px_group; bin++) {
                if (bin < bin_slots.size() && occupied_px_counts.at(bin) <  bin_slots.at(bin).size()) {
                    bin_count = std::max(bin_count, bin_slots.at(bin).at(occupied_px_counts.at(bin)).m_bin_count);
                    pixels.push_back(bin_slots.at(bin).at(occupied_px_counts.at(bin)).m_px);
                    occupied_px_counts.at(bin)++;
                } else {
                    pixels.push_back(PX_grouping::EMPTY_PX);
                }
            }

            ret->append_group(bin_outer, bin_count, pixels);
        }
    }

    return ret;
}

// erase the choosen px
static inline PX_bin_count_pair get_nearest_px_pair_from_vec(int bin_count, std::vector<PX_bin_count_pair>& vec_ref) {
    PX_bin_count_pair ret(PX_grouping::EMPTY_PX, bin_count);
    if (vec_ref.size() == 0)
        return ret;

    int ret_pos = 0;
    ret = vec_ref.at(0);
    int best_diff = bin_count - ret.m_bin_count;  // ensure min abs, better to be positive

    // as the vec_ref are sorted, only the abs is needed

    for (int i = 1; i < vec_ref.size(); i++) {
        int diff = bin_count - vec_ref.at(i).m_bin_count;

        if (best_diff * best_diff > diff * diff) {
            ret_pos = i;
            ret = vec_ref.at(i);
        }
    }

    // erase
    vec_ref.erase(vec_ref.begin() + ret_pos);

    return ret;
}

static inline PX_grouping* get_grouping_greedy_greedy(const std::vector<std::vector<PX_bin_count_pair> >& _bin_slots, int px_group) {
    PX_grouping* ret = new PX_grouping(px_group);

    std::vector<std::vector<PX_bin_count_pair> > bin_slots = _bin_slots;

    for (int bin_outer = 0; bin_outer < bin_slots.size(); bin_outer++) {
        while (bin_slots.at(bin_outer).size() > 0) {
            std::vector<int> pixels;
            int bin_count;

            // erase head
            // int to_erase = 0;
            int to_erase = bin_slots.at(bin_outer).size() - 1;
            pixels.push_back(bin_slots.at(bin_outer).at(to_erase).m_px);
            bin_count = bin_slots.at(bin_outer).at(to_erase).m_bin_count;
            bin_slots.at(bin_outer).erase(bin_slots.at(bin_outer).begin() + to_erase);

            for (int bin = bin_outer + 1; bin < bin_outer + px_group; bin++) {
                if (bin < bin_slots.size()) {
                    PX_bin_count_pair best_pair = get_nearest_px_pair_from_vec(bin_count, bin_slots.at(bin));
                    bin_count = std::max(bin_count, best_pair.m_bin_count);
                    pixels.push_back(best_pair.m_px);
                } else {
                    pixels.push_back(PX_grouping::EMPTY_PX);
                }
            }

            ret->append_group(bin_outer, bin_count, pixels);
        }
    }

    return ret;
}

template <class Element_type>
CSCV_block_statistics* System_matrix_converter_cscv<Element_type>::get_statistics_before_convertion(const CSC_matrix<Element_type>* csc, Block<Element_type>* block) {
    int block_angle_count = block->m_angle_count;
    int block_pixel_count = block->m_size_image_x * block->m_size_image_y;

    int angle_bin_lhs[block_angle_count], angle_bin_rhs[block_angle_count];
    int pixel_angle_bin_lhs[block_angle_count], pixel_angle_bin_rhs[block_angle_count];
    int ref_pixel_angle_bin_lhs[block_angle_count], ref_pixel_angle_bin_rhs[block_angle_count];
    Result_array<float> nnz_bin_in_angle, nnz_bin_per_angle_in_pixel;

    CSCV_block_statistics* ret = new CSCV_block_statistics;
    ret->m_pixel_bin_count = new Dense_vector<int>(block_pixel_count);
    ret->m_pixel_bin_lhs_shift = new Dense_vector<int>(block_pixel_count);

    // int mid_img_x = ((m_x_group_size + 1) / 2) - 1;
    // int mid_img_y = ((m_y_group_size + 1) / 2) - 1;
    // int ref_pixel_id = m_x_group_size * mid_img_y + mid_img_x;

    int ref_px_img_x = ((m_x_group_size + 1) / 2) - 1;
    int ref_px_img_y = ((m_y_group_size + 1) / 2) - 1;

    if (m_comp_cfg.m_ref_px_x_offset != -1)
        ref_px_img_x = m_comp_cfg.m_ref_px_x_offset;
    if (m_comp_cfg.m_ref_px_y_offset != -1)
        ref_px_img_y = m_comp_cfg.m_ref_px_y_offset;

    ASSERT_AND_PRINTF(ref_px_img_x >= 0 && ref_px_img_x < m_x_group_size, "");
    ASSERT_AND_PRINTF(ref_px_img_y >= 0 && ref_px_img_y < m_y_group_size, "");

    int ref_pixel_id = m_x_group_size * ref_px_img_y + ref_px_img_x;

    std::string log_filename = strprintf("block_x_%d_y_%d_a_%d.log", block->m_start_x, block->m_start_y, (int)(block->m_start_angle * m_img_param.m_delta_angle));

    if (Flag::get_instance().dump_figure())
        Logger::get_instance().set_filename(log_filename);

    get_pixel_bin_ranges(csc, ref_pixel_id, ref_pixel_angle_bin_lhs, ref_pixel_angle_bin_rhs);

    if (Flag::get_instance().dump_figure()) {
        Logger::get_instance().write("ref px's <lhs, rhs>:\n");
        Logger::get_instance().write(arrs_to_string(block_angle_count, ref_pixel_angle_bin_lhs, ref_pixel_angle_bin_rhs) + "\n");
    }

    int cscv_vec_count = 0;

    Dense_vector<int> px_nnz_count(m_y_group_size * m_x_group_size);

    // traverse all pxs, and build bin ranges
    for (int block_img_y = 0; block_img_y < m_y_group_size; block_img_y++) {
        for (int block_img_x = 0; block_img_x < m_x_group_size; block_img_x++) {
            int col = block_img_y * m_x_group_size + block_img_x;
            bool is_edge = block_img_x == 0 || block_img_x == m_x_group_size - 1 || block_img_y == 0 || block_img_y == m_y_group_size - 1;
            bool is_corner = (block_img_x == 0 || block_img_x == m_x_group_size - 1) && (block_img_y == 0 || block_img_y == m_y_group_size - 1);
            bool is_ref = (col == ref_pixel_id);
            int max_nnz_bin_in_angle = 0;

            px_nnz_count.at(col) = csc->m_col_offsets[col + 1] - csc->m_col_offsets[col];

            get_pixel_bin_ranges(csc, col, &pixel_angle_bin_lhs[0], &pixel_angle_bin_rhs[0]);

            if ((is_corner || is_ref) && Flag::get_instance().dump_figure()) {
                Logger::get_instance().write(strprintf("\npx x = %d, y = %d before:\n", block_img_x, block_img_y));
                Logger::get_instance().write(arrs_to_string(block_angle_count, pixel_angle_bin_lhs, pixel_angle_bin_rhs) + "\n");
            }

            ret->m_pixel_bin_lhs_shift->at(col) = pad_lhs_and_get_bin_shift(ref_pixel_angle_bin_lhs, pixel_angle_bin_lhs, block_angle_count);

            // statistics
            for (int angle = 0; angle < block_angle_count; angle++) {
                int bin_count = pixel_angle_bin_rhs[angle] - pixel_angle_bin_lhs[angle] + 1;
                max_nnz_bin_in_angle = std::max(max_nnz_bin_in_angle, bin_count);
                nnz_bin_in_angle.append_result(bin_count);
            }

            // fix
            for (int angle = 0; angle < block_angle_count; angle++) {
                pixel_angle_bin_rhs[angle] = pixel_angle_bin_lhs[angle] + max_nnz_bin_in_angle - 1;
            }

            if ((is_corner || is_ref) && Flag::get_instance().dump_figure()) {
                Logger::get_instance().write(strprintf("\npx x = %d, y = %d after, bin count = %d:\n", block_img_x, block_img_y, max_nnz_bin_in_angle));
                Logger::get_instance().write(arr_to_string(block_angle_count, pixel_angle_bin_lhs) + "\n");
            }

            if (block_img_x == 0 && block_img_y == 0) {
                // copy the lhs and rhs
                memcpy(angle_bin_lhs, pixel_angle_bin_lhs, sizeof(int) * block_angle_count);
                memcpy(angle_bin_rhs, pixel_angle_bin_rhs, sizeof(int) * block_angle_count);
            } else {
                for (int angle = 0; angle < block_angle_count; angle++) {
                    angle_bin_lhs[angle] = std::min(angle_bin_lhs[angle], pixel_angle_bin_lhs[angle]);
                    angle_bin_rhs[angle] = std::max(angle_bin_rhs[angle], pixel_angle_bin_rhs[angle]);
                }
            }

            ret->m_pixel_bin_count->at(col) = max_nnz_bin_in_angle;

            nnz_bin_per_angle_in_pixel.append_result(max_nnz_bin_in_angle);

            cscv_vec_count += max_nnz_bin_in_angle;
        }
    }

    ret->m_nnz_naive = cscv_vec_count * m_angle_group_size;
    ret->m_nnz_native = csc->m_nz_count;

    ret->m_ref_pixel_lhs_bin_id = ref_pixel_angle_bin_lhs[0] - angle_bin_lhs[0];
    ret->m_block_angle_bin_count = angle_bin_rhs[0] - angle_bin_lhs[0] + 1;

    if (Flag::get_instance().dump_figure()) {
        Logger::get_instance().write(strprintf("\nblock, bin count = %d, lhs:\n", ret->m_block_angle_bin_count));
        Logger::get_instance().write(arr_to_string(block_angle_count, angle_bin_lhs) + "\n");
    }

    // check the range consistency
    ret->m_angle_bin_starts = new Dense_vector<int>(block_angle_count);
    for (int angle = 0; angle < block_angle_count; angle++) {
        ret->m_angle_bin_starts->at(angle) = angle_bin_lhs[angle];
        ASSERT_AND_PRINTF(angle_bin_rhs[angle] - angle_bin_lhs[angle] + 1 == ret->m_block_angle_bin_count, "size %d != %d\n",
                          angle_bin_rhs[angle] - angle_bin_lhs[angle] + 1, ret->m_block_angle_bin_count);

        // double check the alignment of block lhs and ref lhs
        ASSERT_AND_PRINTF(ref_pixel_angle_bin_lhs[angle] - angle_bin_lhs[angle] == ref_pixel_angle_bin_lhs[0] - angle_bin_lhs[0],
                          "lhs alignment mismatch at angle %d, %d - %d != %d - %d\n", angle,
                          ref_pixel_angle_bin_lhs[angle], angle_bin_lhs[angle], ref_pixel_angle_bin_lhs[0], angle_bin_lhs[0]);

        ASSERT_AND_PRINTF(angle_bin_lhs[angle] >= 0, "angle bin lhs %d is less than 0!\n",
                          angle_bin_lhs[angle]);
        // ASSERT_AND_PRINTF(angle_bin_lhs[angle] + ret->m_block_angle_bin_count < m_img_param.m_num_bin, "angle bin rhs excceed the range! %d + %d >= %d\n",
        //                   angle_bin_lhs[angle], ret->m_block_angle_bin_count, m_img_param.m_num_bin);
    }

    // fix lhs, and make sure that lhs shift starts from 0
    for (int px = 0; px < block_pixel_count; px++) {
        ret->m_pixel_bin_lhs_shift->at(px) += (ref_pixel_angle_bin_lhs[0] - angle_bin_lhs[0]);
        ASSERT_AND_PRINTF(ret->m_pixel_bin_lhs_shift->at(px) >= 0, "%d <= 0\n", ret->m_pixel_bin_lhs_shift->at(px));
    }

    // render data for figure 1
    if (Flag::get_instance().dump_figure()) {
        Logger::get_instance().write("\npx <bin_off, bin_count, pad zero>:\n");
        for (int block_img_y = 0; block_img_y < m_y_group_size; block_img_y++) {
            for (int block_img_x = 0; block_img_x < m_x_group_size; block_img_x++) {
                int px = block_img_y * m_x_group_size + block_img_x;
                Logger::get_instance().write(strprintf("<%d, %d, %d>\t", ret->m_pixel_bin_lhs_shift->at(px), ret->m_pixel_bin_count->at(px),
                                                                        m_angle_group_size * ret->m_pixel_bin_count->at(px) - px_nnz_count.at(px)));
            }
            Logger::get_instance().write("\n");
        }
    }

    std::vector<std::vector<PX_bin_count_pair> > pb_vec;

    for (int bin = 0; bin < ret->m_block_angle_bin_count; bin++)
        pb_vec.emplace_back();

    // insert <px_id, bin_count> into slots that ordered by bin_off
    for (int px = 0; px < block_pixel_count; px++) {
        ASSERT_AND_PRINTF(ret->m_pixel_bin_lhs_shift->at(px) < ret->m_block_angle_bin_count, "");
        pb_vec.at(ret->m_pixel_bin_lhs_shift->at(px)).emplace_back(px, ret->m_pixel_bin_count->at(px));
    }

    ASSERT_AND_PRINTF(get_effective_bin_count(pb_vec) == block_pixel_count, "px mismatch %d and %d\n", get_effective_bin_count(pb_vec), block_pixel_count);

    // sort each slots by bin_count
    for (int bin = 0; bin < ret->m_block_angle_bin_count; bin++) {
        if (pb_vec.at(bin).size() > 1) {
            std::sort(pb_vec.at(bin).begin(), pb_vec.at(bin).end(), [](const PX_bin_count_pair& l, const PX_bin_count_pair& r) {
                if (l.m_bin_count < r.m_bin_count)
                    return true;
                if (l.m_bin_count > r.m_bin_count)
                    return false;
                return l.m_px < r.m_px;
            });
        }
    }

    // print px ordered by bin count

    if (Flag::get_instance().dump_figure()) {
        Logger::get_instance().write("\n[<px id, bin count>] ordered by bin off\n");

        for (int bin = 0; bin < ret->m_block_angle_bin_count; bin++) {
            std::stringstream ss;
            ss << "bin_off = " << bin << "[";
            for (auto pb_pair : pb_vec.at(bin)) {
                ss << "<" << pb_pair.m_px << "  " << pb_pair.m_bin_count << ">, ";
            }
            ss << "]\n";
            Logger::get_instance().write(ss.str());
        }
    }


    ASSERT_AND_PRINTF(get_effective_bin_count(pb_vec) == block_pixel_count, "px mismatch %d and %d\n", get_effective_bin_count(pb_vec), block_pixel_count);

    // build pxg from the sorted slots by a given policy
    ret->m_pxgs = get_grouping_greedy_greedy(pb_vec, block->m_pxg_size);
    #pragma omp critical
    {
        uint8_t tmp_max = m_max_pxg_bin_count;
        m_max_pxg_bin_count = std::max(tmp_max, ret->m_pxgs->get_max_bin_count());
    }

    int last_printed_bin_offset = -1;

    if (Flag::get_instance().dump_figure()){
        Logger::get_instance().write("\npxg [<pxs, bin count>] ordered by bin off\n");

        // print px grouping ordered by bin_off (for figures)
        for (int pxg = 0; pxg < ret->m_pxgs->get_group_count(); pxg++) {
            int bin_offset = ret->m_pxgs->get_bin_offs().at(pxg);

            if (bin_offset != last_printed_bin_offset) {
                Logger::get_instance().write(strprintf("\nbin_off = %d: ", bin_offset));
                last_printed_bin_offset = bin_offset;
            }

            // <px, cnt>
            std::stringstream ss;
            ss << "[";
            bool mismatch_occured = false;
            int first_bin_count;

            for (int px_offset = 0; px_offset < block->m_pxg_size; px_offset++) {
                int px_id = ret->m_pxgs->get_pixels().at(pxg * block->m_pxg_size + px_offset);
                int bin_count = -1;
                if (px_id != PX_grouping::EMPTY_PX) {
                    bin_count = ret->m_pixel_bin_count->at(px_id);
                }
                ss << "<" << px_id << "  " << bin_count << ">, ";

                if (px_offset == 0) {
                    first_bin_count = bin_count;
                } else {
                    if (bin_count != first_bin_count) {
                        mismatch_occured = true;
                    }
                }
            }
            if (mismatch_occured) {
                ss << "***";
            }

            ss << "], ";
            Logger::get_instance().write(ss.str());
        }
    }

    ASSERT_AND_PRINTF(ret->m_pxgs->get_max_bin_count() <= c_pxg_max_bin, "bin count excceed! %d > %d\n",
                      ret->m_pxgs->get_max_bin_count(), c_pxg_max_bin);

    ASSERT_AND_PRINTF(ret->m_pxgs->get_effective_px_count() == block_pixel_count,
                     "tea grouping do not contains all px, %d != %d\n", ret->m_pxgs->get_effective_px_count(), block_pixel_count);

    ret->m_nnz_tea = ret->m_pxgs->get_bin_full() * block->m_angle_group_size;
    ret->m_vec_tea = ret->m_pxgs->get_bin_full();

    if (m_angle_group_size <= 8) {
        ret->m_mask_granularity = 1;
    } else if (m_angle_group_size == 16) {
        ret->m_mask_granularity = 2;
    } else {
        ASSERT_AND_PRINTF(false, "unacceptable angle group size %d\n", m_angle_group_size);
    }
    ret->m_masks_bytes = ret->m_vec_tea * ret->m_mask_granularity;
    ret->m_popcnts_size = ret->m_vec_tea;

    auto bin_offs = ret->m_pxgs->get_bin_offs();
    auto bin_counts = ret->m_pxgs->get_bin_counts();
    auto pixels = ret->m_pxgs->get_pixels();

    // insert pxgs <px_id, bin_off> to slots ordered by bin_count
    for (int pxg = 0; pxg < ret->m_pxgs->get_group_count(); pxg++) {
        int bin_count = bin_counts.at(pxg);
        int bin_off = bin_offs.at(pxg);

        ret->m_tea_slots[bin_count].m_bin_offs.push_back(bin_off);

        int pxg_px_offset = pxg * block->m_pxg_size;

        for (int px_offset = pxg_px_offset; px_offset < pxg_px_offset + block->m_pxg_size; px_offset++) {
            ret->m_tea_slots[bin_count].m_pixels.push_back(pixels.at(px_offset));
        }
    }

    if (Flag::get_instance().dump_figure()) {
        Logger::get_instance().write("\npxg [<pxs, bin offset>] ordered by bin count\n");

        // print the figure of <px_id, bin_off> slots ordered by bin_count
        for(auto p : ret->m_tea_slots) {
            Logger::get_instance().write(strprintf("\nbin count = %d: ", p.first));
            for (int i = 0; i < p.second.m_bin_offs.size(); i++) {
                std::stringstream ss;
                ss << "<px: [";
                for (int px_offset = i * block->m_pxg_size; px_offset < (i + 1) * block->m_pxg_size; px_offset++) {
                    ss << p.second.m_pixels.at(px_offset) << ",";
                }
                ss << "], off: " << p.second.m_bin_offs.at(i) << ">, ";
                Logger::get_instance().write(ss.str());
            }
        }
        Logger::get_instance().write("\n");
    }


    // finally, force pad bin for yt buffer
    ret->m_block_angle_bin_count = div_and_ceil(ret->m_block_angle_bin_count, m_angle_group_size) * m_angle_group_size;

    m_nnz += csc->m_nz_count;
    m_cscv_element_count += ret->m_nnz_naive;
    m_tea_nnz += ret->m_nnz_tea;

    Logger::get_instance().close_file();

    return ret;
}

// In cscvb, only global lhs is needed (although local lhs may be used)
template <class Element_type>
void System_matrix_converter_cscv<Element_type>::convert_system_matrix_to_cscvb(const CSC_matrix<Element_type>* csc, Block<Element_type>* block,
                                                                                CSCV_block_mem_pool<Element_type>* pool, CSCV_block_statistics* sta) {
    int block_angle_count = block->m_angle_count;

    CSCVB_matrix_block<Element_type>* ret = new CSCVB_matrix_block<Element_type>;
    block->m_cscvb_block = ret;

    ret->m_pixel_bin_count = new Dense_vector<int>(m_x_group_size * m_y_group_size, &sta->m_pixel_bin_count->at(0));

    ret->m_size_image_x = m_x_group_size;
    ret->m_size_image_y = m_y_group_size;
    ret->m_angle_group_size = m_angle_group_size;

    ret->m_block_angle_bin_count = sta->m_block_angle_bin_count;

    ret->m_angle_bin_starts = new Dense_vector<int>(block_angle_count, &sta->m_angle_bin_starts->at(0));

    ret->m_pxg_count = sta->m_pxgs->get_group_count();
    ret->m_pxg_size = block->m_pxg_size;

    // fetch offsets from block
    auto bin_offs = sta->m_pxgs->get_bin_offs();
    auto bin_counts = sta->m_pxgs->get_bin_counts();
    auto pixels = sta->m_pxgs->get_pixels();


    ret->m_tea_group_offs[0] = 0;
    int tea_data_offset = 0, tea_px_offset = 0, tea_bin_offs_offset = 0;
    int cat_mask_byte_offset = 0, cat_data_offset = 0, cat_popcnts_offset = 0;
    int popcnt_sum = 0;

    if (!(m_comp_cfg.m_run_cscvb_cat || m_comp_cfg.m_run_cscvb_tea))
        return;

    // build tea
    ret->m_tea_pxs = pool->get_next_tea_pxs(pixels.size());
    ret->m_tea_bin_offs = pool->get_next_tea_bin_offs(bin_offs.size());
    ret->m_tea_data = pool->get_next_tea_data(sta->m_nnz_tea);

    // build cat
    // if (m_comp_cfg.m_run_cscvb_cat) {
        ret->m_cat_data = pool->get_next_cat_data(sta->m_nnz_native);
        ret->m_cat_masks = pool->get_next_cat_masks(sta->m_masks_bytes);
        ret->m_cat_popcnts = pool->get_next_cat_popcnts(sta->m_popcnts_size);
    // }

    for (uint8_t bin_count = 0; bin_count < c_pxg_max_bin; bin_count++) {
        if (sta->m_tea_slots.count(bin_count) == 0) {
            ret->m_tea_group_offs[bin_count + 1] = ret->m_tea_group_offs[bin_count];
        } else {
            auto slot = sta->m_tea_slots.at(bin_count);

            // iterate each vxg
            for (int i = 0; i < slot.m_bin_offs.size(); i++) {
                Dense_vector<Element_type> tea_data(bin_count * ret->m_pxg_size * m_angle_group_size, &ret->m_tea_data->at(tea_data_offset));

                Dense_vector<uint32_t> tmp_mask(bin_count * ret->m_pxg_size);
                tmp_mask.set_zero();

                // fill data in pxg
                int pxg_px_offset = i * ret->m_pxg_size;
                for (int px_offset = pxg_px_offset; px_offset < pxg_px_offset + ret->m_pxg_size; px_offset++) {
                    if (slot.m_pixels.at(px_offset) != PX_grouping::EMPTY_PX) {
                        int col = slot.m_pixels.at(px_offset);
                        int block_img_x = col % m_x_group_size;
                        int block_img_y = col / m_x_group_size;

                        for (int offset = csc->m_col_offsets[col]; offset < csc->m_col_offsets[col + 1]; offset++) {
                            int row = csc->m_row_idxs[offset];
                            int angle_id_in_block = row / m_img_param.m_num_bin;
                            ASSERT_AND_PRINTF(angle_id_in_block < m_angle_group_size, "%d >= %d, this cannot happen!\n", angle_id_in_block, m_angle_group_size);
                            int bin_id = row % m_img_param.m_num_bin;
                            int pixel_bin_id = bin_id - (sta->m_angle_bin_starts->at(angle_id_in_block) + sta->m_pixel_bin_lhs_shift->at(col));

                            Element_type val = csc->m_vals[offset];

                            int tea_inner_offset = angle_id_in_block + pixel_bin_id * ret->m_pxg_size * m_angle_group_size
                                                                                    + (px_offset - pxg_px_offset) * m_angle_group_size;

                            ASSERT_AND_PRINTF(tea_data.at(tea_inner_offset) == 0, "not zero! %f, img_x = %d, img_y = %d, angle = %d, bin = <abs: %d, local: %d>\n",
                                            tea_data.at(tea_inner_offset), block_img_x, block_img_y, angle_id_in_block, bin_id, pixel_bin_id);
                            tea_data.at(tea_inner_offset) = val;

                            // TODO: Remove this when running tea only
                            uint32_t& mask_ref = tmp_mask.at((px_offset - pxg_px_offset) + pixel_bin_id * ret->m_pxg_size);

                            uint32_t mask_check = mask_ref & (1 << angle_id_in_block);
                            ASSERT_AND_PRINTF(mask_check == 0, "this bit has already been marked! %u\n", mask_check);

                            mask_ref |= 1 << angle_id_in_block;
                        }

                        ret->m_tea_pxs->at(px_offset + tea_px_offset) = col;
                    } else {
                        ret->m_tea_pxs->at(px_offset + tea_px_offset) = m_x_group_size * m_y_group_size;  // more 1px out of the block
                    }
                }

                ret->m_tea_bin_offs->at(tea_bin_offs_offset++) = slot.m_bin_offs.at(i);  // the start bin of this px group

                tea_data_offset += bin_count * ret->m_pxg_size * m_angle_group_size;

                if (!m_comp_cfg.m_run_cscvb_cat)
                    continue;

                // copy vals according to the mask
                for (int bin_off0 = 0; bin_off0 < bin_count; bin_off0++) {
                    for (int px_off0 = 0; px_off0 < ret->m_pxg_size; px_off0++) {
                        const uint32_t& mask_ref = tmp_mask.at(px_off0 + bin_off0 * ret->m_pxg_size);
                        for (int angle_off0 = 0; angle_off0 < m_angle_group_size; angle_off0++) {
                            int tea_inner_offset = angle_off0 + bin_off0 * ret->m_pxg_size * m_angle_group_size + px_off0 * m_angle_group_size;
                            if (mask_ref & (1 << angle_off0)) {
                                ret->m_cat_data->at(cat_data_offset++) = tea_data.at(tea_inner_offset);
                            } else {
                                ASSERT_AND_PRINTF(tea_data.at(tea_inner_offset) == 0, "");
                            }
                        }
                    }
                }

                // after building the tea data, fetch cat data by masks from the tea data

                // copy the mask, and also get the popcnt for further validation
                Dense_vector<std::byte> mask_dst_container(sta->m_mask_granularity * bin_count * ret->m_pxg_size, &ret->m_cat_masks->at(cat_mask_byte_offset));
                Dense_vector<uint8_t> popcnts_dst(bin_count * ret->m_pxg_size, &ret->m_cat_popcnts->at(cat_popcnts_offset));
                if (sta->m_mask_granularity == 1) {
                    Dense_vector<uint8_t> mask_dst(&mask_dst_container);
                    ASSERT_AND_PRINTF(tmp_mask.get_size() == mask_dst.get_size(), "");
                    for (int off = 0; off < mask_dst.get_size(); off++) {
                        mask_dst.at(off) = tmp_mask.at(off);
                        ASSERT_AND_PRINTF(tmp_mask.at(off) < 1 << (sta->m_mask_granularity * 8), "");
                        int popcnt = __builtin_popcount(tmp_mask.at(off));
                        popcnt_sum += popcnt;
                        popcnts_dst.at(off) = popcnt;
                    }
                } else if (sta->m_mask_granularity == 2) {
                    Dense_vector<uint16_t> mask_dst(&mask_dst_container);
                    ASSERT_AND_PRINTF(tmp_mask.get_size() == mask_dst.get_size(), "");
                    for (int off = 0; off < mask_dst.get_size(); off++) {
                        mask_dst.at(off) = tmp_mask.at(off);
                        ASSERT_AND_PRINTF(tmp_mask.at(off) < 1 << (sta->m_mask_granularity * 8), "");
                        int popcnt = __builtin_popcount(tmp_mask.at(off));
                        popcnt_sum += popcnt;
                        popcnts_dst.at(off) = popcnt;
                    }
                } else {
                    ASSERT_AND_PRINTF(false, "unsupported mask size %d\n", sta->m_mask_granularity);
                }

                cat_mask_byte_offset += bin_count * ret->m_pxg_size * sta->m_mask_granularity;
                cat_popcnts_offset += bin_count * ret->m_pxg_size;
            }  // bin offsets
            tea_px_offset += ret->m_pxg_size * slot.m_bin_offs.size();
            ret->m_tea_group_offs[bin_count + 1] = ret->m_tea_group_offs[bin_count] + slot.m_bin_offs.size();
        }  // start bins
    }

    if (m_comp_cfg.m_run_cscvb_cat) {
        ASSERT_AND_PRINTF(cat_mask_byte_offset == ret->m_cat_masks->get_bytes(), "");
        ASSERT_AND_PRINTF(cat_data_offset == sta->m_nnz_native, "");
        ASSERT_AND_PRINTF(popcnt_sum == ret->m_cat_data->get_size(), "");
    }
}
