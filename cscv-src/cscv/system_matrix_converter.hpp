#pragma once

#include <set>
// #include <tbb/atomic.h>

#include "base/logger.hpp"
#include "base/result_array.hpp"
#include "cscv/partition.hpp"
#include "ct/system_matrix.hpp"
#include "data/data_container.hpp"

#include <atomic>

struct PX_bin_count_pair {
    int m_px;
    int m_bin_count;

    PX_bin_count_pair(int px, int bin_count) : m_px(px), m_bin_count(bin_count) {}
};


struct PXG_slot {
    std::vector<int> m_pixels;
    std::vector<int> m_bin_offs;
};

class PX_grouping {
private:
    std::vector<int> m_pixels;  // notice that the px id can be replaced to uint16_t, or even uint8_t when using 16 * 16 kernel
    std::vector<int> m_bin_offs;
    std::vector<int> m_bin_counts;
    int m_group_size;

public:
    PX_grouping(int group_size) : m_group_size(group_size) {
        ASSERT_AND_PRINTF(group_size > 0 && group_size <= 64, "unacceptable group size %d\n", group_size);
    }

    void append_group(int bin_off, int bin_count, const std::vector<int>& pixels) {
        m_bin_offs.push_back(bin_off);
        m_bin_counts.push_back(bin_count);

        ASSERT_AND_PRINTF(pixels.size() == m_group_size, "group size mismatch! %lu != %d\n", pixels.size(), m_group_size);
        m_pixels.insert(m_pixels.end(), pixels.begin(), pixels.end());
    }

    int get_group_count() const { return m_bin_offs.size(); }

    const std::vector<int>& get_bin_offs() const { return m_bin_offs; }
    const std::vector<int>& get_bin_counts() const { return m_bin_counts; }
    const std::vector<int>& get_pixels() const { return m_pixels; }

    uint8_t get_max_bin_count() const {
        int ret = 0;
        for (auto bin_count : m_bin_counts)
            ret = std::max(ret, bin_count);
        return ret;
    }

    int get_bin_full() const {
        int ret = 0;
        for (int i = 0; i < m_bin_offs.size(); i++) {
            ret += m_bin_counts.at(i);  // why write the code like this??
        }
        ret *= m_group_size;
        return ret;
    }

    // debug
    int get_effective_px_count() const {
        std::set<int> px_ids;

        for (auto pixel : m_pixels) {
            if (pixel != EMPTY_PX) {
                ASSERT_AND_PRINTF(px_ids.count(pixel) == 0, "pixel %d found twice!\n", pixel);
                px_ids.insert(pixel);
            }
        }

        return px_ids.size();
    }

    static constexpr int EMPTY_PX = -233;
};

/** 
 * created before building cscv data.
 * 
 **/
struct CSCV_block_statistics {
    // ct param
    int m_angle_group_size;

    // nnz
    size_t m_nnz_naive = 0;  // padded, m_naive_vec_count * t_vec_angle
    size_t m_nnz_native = 0;  // no zero padding, just copied from csc

    // basic
    int m_block_angle_bin_count;
    int m_ref_pixel_lhs_bin_id;
    Dense_vector<int>* m_pixel_bin_count = nullptr;
    Dense_vector<int>* m_pixel_bin_lhs_shift = nullptr;  // compared with ref lhs. this can easily be compressed, compared with bin_start
    Dense_vector<int>* m_angle_bin_starts = nullptr;  // use this as lhs

    PX_grouping* m_pxgs = nullptr;

    std::map<int, PXG_slot> m_tea_slots;

    size_t m_nnz_tea = 0;

    size_t m_vec_tea = 0;
    size_t m_masks_bytes = 0;
    size_t m_popcnts_size = 0;
    uint8_t m_mask_granularity = 0;

    void free_before_computation() {
        if (m_pxgs) delete m_pxgs;
        m_pxgs = nullptr;

        m_tea_slots = std::map<int, PXG_slot>();

        if (m_pixel_bin_count) delete m_pixel_bin_count;
        m_pixel_bin_count = nullptr;
        if (m_pixel_bin_lhs_shift) delete m_pixel_bin_lhs_shift;
        m_pixel_bin_lhs_shift = nullptr;
        // if (m_angle_bin_starts) delete m_angle_bin_starts;
        // m_angle_bin_starts = nullptr;
    }

    ~CSCV_block_statistics() {
        if (m_pixel_bin_count) delete m_pixel_bin_count;
        if (m_pixel_bin_lhs_shift) delete m_pixel_bin_lhs_shift;
        if (m_angle_bin_starts) delete m_angle_bin_starts;
        if (m_pxgs) delete m_pxgs;
    }
};

template <class Element_type>
class CSCV_block_mem_pool {
    Dense_vector<Element_type>* m_tea_data_pool = nullptr;
    size_t m_tea_data_allocated = 0, m_tea_data_size = 0;

    Dense_vector<uint16_t>* m_tea_pxs_pool = nullptr;
    size_t m_tea_pxs_size = 0, m_tea_pxs_allocated = 0;
    Dense_vector<uint8_t>* m_tea_bin_offs_pool = nullptr;
    size_t m_tea_bin_offs_size = 0, m_tea_bin_offs_allocated = 0;

    Dense_vector<Element_type>* m_cat_data_pool = nullptr;
    size_t m_cat_data_size = 0, m_cat_data_allocated = 0;
    Dense_vector<std::byte>* m_cat_masks_pool = nullptr;
    size_t m_cat_masks_size = 0, m_cat_masks_allocated = 0;
    Dense_vector<uint8_t>* m_cat_popcnts_pool = nullptr;
    size_t m_cat_popcnts_size = 0, m_cat_popcnts_allocated = 0;

public:
    CSCV_block_mem_pool(const std::vector<CSCV_block_statistics*>& stas);  // only use statistics in it

    Dense_vector<Element_type>* get_next_tea_data(size_t size);
    Dense_vector<uint16_t>* get_next_tea_pxs(size_t size);
    Dense_vector<uint8_t>* get_next_tea_bin_offs(size_t size);

    Dense_vector<Element_type>* get_next_cat_data(size_t size);
    Dense_vector<std::byte>* get_next_cat_masks(size_t bytes);
    Dense_vector<uint8_t>* get_next_cat_popcnts(size_t size);

    size_t get_tea_data_size() const { return m_tea_data_size; }
    size_t get_tea_pxs_size() const { return m_tea_pxs_size; }
    size_t get_tea_bin_offs_size() const { return m_tea_bin_offs_size; }

    size_t get_cat_data_size() const { return m_cat_data_size; }
    size_t get_cat_masks_bytes() const { return m_cat_masks_size; }
    size_t get_cat_popcnts_size() const { return m_cat_popcnts_size; }

    void free_tea() {
        if (m_tea_data_pool) delete m_tea_data_pool;
        // if (m_tea_pxs_pool) delete m_tea_pxs_pool;
        // if (m_tea_bin_offs_pool) delete m_tea_bin_offs_pool;
        m_tea_data_pool = nullptr;
        // m_tea_pxs_pool = nullptr;
        // m_tea_bin_offs_pool = nullptr;
    }

    void free_cat() {
        if (m_cat_data_pool) delete m_cat_data_pool;
        m_cat_data_pool = nullptr;
        if (m_cat_masks_pool) delete m_cat_masks_pool;
        m_cat_masks_pool = nullptr;
        if (m_cat_popcnts_pool) delete m_cat_popcnts_pool;
        m_cat_popcnts_pool = nullptr;
    }

    ~CSCV_block_mem_pool() {
        if (m_tea_data_pool) delete m_tea_data_pool;
        if (m_tea_pxs_pool) delete m_tea_pxs_pool;
        if (m_tea_bin_offs_pool) delete m_tea_bin_offs_pool;
        if (m_cat_data_pool) delete m_cat_data_pool;
        if (m_cat_masks_pool) delete m_cat_masks_pool;
        if (m_cat_popcnts_pool) delete m_cat_popcnts_pool;
    }
};

/**
 * The system matrix is generated part by part.
 * Thus, the convertion is to generate blocks in the part of system matrix.
 * The block params are already setup in the partition stage.
 * 
 * This class will also do some statistics during the convertion process, like the extra NNZ, and possibly COST computation.
 * 
 * Now, this class also in charge of the generation of coo.
 * 
 * TODO: This class should not be a template class, just let the convertion function a template function.
 **/
template <class Element_type>
class System_matrix_converter_cscv {
public:
    System_matrix_converter_cscv(const Img_param& img_param, int x_group_size, int y_group_size, int angle_group_size, Computation_config comp_cfg) :
                                m_img_param(img_param), m_x_group_size(x_group_size), m_y_group_size(y_group_size),
                                m_angle_group_size(angle_group_size),  m_comp_cfg(comp_cfg) {
        m_nnz = 0;
        m_cscv_element_count = 0;
        m_tea_nnz = 0;
        m_max_pxg_bin_count = 0;
    }

private:
    // configuration
    static constexpr int m_alignment = MM_DEFAULT_ALIGNMENT;  // 64B

    int m_x_group_size, m_y_group_size, m_angle_group_size;

    Img_param m_img_param;
    Computation_config m_comp_cfg;

    // statistics
    int m_extra_nnz_counter;

    // std::vector<std::vector<COO_matrix_buffer*> > m_system_matrixes;  // for blocks in parts

    std::atomic<uint64_t> m_nnz = 0, m_cscv_element_count = 0, m_tea_nnz = 0;
    std::atomic<uint8_t> m_max_pxg_bin_count = 0;

    void get_pixel_bin_ranges(const CSC_matrix<Element_type>* csc, int pixel_id, int* angle_bin_lhs, int* angle_bin_rhs);
    static int pad_lhs_and_get_bin_shift(const int* ref_lhs, int* target_lhs, int block_angle_count);  // return the shift relative to ref lhs

public:
    void clear_statictics();

    CSCV_block_statistics* get_statistics_before_convertion(const CSC_matrix<Element_type>* csc, Block<Element_type>* block);

    // CSCV_block_statistics get_cscv_statistics
    void convert_system_matrix_to_cscvb(const CSC_matrix<Element_type>* csc, Block<Element_type>* block,
                                        CSCV_block_mem_pool<Element_type>* pool, CSCV_block_statistics* sta);

    void get_nnz_expansion_ratio() const {
        uint64_t cscv_element_count = m_cscv_element_count;
        uint64_t nnz = m_nnz;
        uint64_t tea_nnz = m_tea_nnz;

        double cscv_expansion = cscv_element_count * 1.0 / nnz;
        double tea_expansion = tea_nnz * 1.0 / nnz;
        double tea_cscv_expansion = tea_nnz * 1.0 / cscv_element_count;

        double ori_estimated_min_time = (nnz * sizeof(Element_type)) / (1024.0 * 1024.0 * 1024.0) / 208.0;
        printf("[CSCV converter] cscv naive nnz = %lu, min time = %f (208 GB/s), tea nnz = %lu, original nnz = %lu\n",
                cscv_element_count, ori_estimated_min_time, tea_nnz, nnz);
        printf("[CSCV converter] cscv expansion = %f, tea expansion = %f, tea cscv expansion = %f\n", cscv_expansion, tea_expansion, tea_cscv_expansion);
    }

    uint64_t get_original_nnz() const { return m_nnz; }
    uint64_t get_tea_nnz() const { return m_tea_nnz; }

    double get_cscv_expansion() const {return m_cscv_element_count * 1.0 / m_nnz; }

    uint8_t get_max_pxg_bin_count() const { return m_max_pxg_bin_count; }
};

#include "system_matrix_converter.tcc"
