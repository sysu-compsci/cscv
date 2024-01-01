#include "partition.hpp"

template <class Element_type>
Block<Element_type>::~Block() {
    if (m_cscvb_block)
        delete m_cscvb_block;
    if (m_csc_block)
        delete m_csc_block;
    if (m_coo_block)
        delete m_coo_block;
    if (m_cscv_sta)
        delete m_cscv_sta;
}

template <class Element_type>
Part<Element_type>::~Part() {
    if (m_part_image)
        delete m_part_image;
    if (m_part_detector)
        delete m_part_detector;

    if (m_cscv_mem_pool)
        delete m_cscv_mem_pool;

    for (Block<Element_type>* block : m_blocks)
        delete block;

    if (m_csc_part)
        delete m_csc_part;
    if (m_csr_part)
        delete m_csr_part;
    if (m_coo_part)
        delete m_coo_part;

    if (m_tea_soa_context) delete m_tea_soa_context;

    if (m_y_lhss) delete m_y_lhss;
    if (m_y_rhss) delete m_y_rhss;
}

template <class Element_type>
bool Part<Element_type>::is_empty() const {
    if (m_size_image_x == 0 || m_size_image_y == 0 || m_angle_count == 0)
        return true;
    return false;
}

template <class Element_type>
size_t Part<Element_type>::get_blocks_coo_nnz() const {
    size_t ret = 0;
    for (const Block<Element_type>* block : m_blocks) {
        ASSERT_AND_PRINTF(block->m_coo_block != nullptr, "");
        ret += block->m_coo_block->m_nz_count;
    }
    return ret;
}

template <class Element_type>
void Part<Element_type>::build_cscvb_yt_buffer() {
    int max_bin = 0;

    for (Block<Element_type>* block : m_blocks) {
        max_bin = std::max(max_bin, block->m_cscvb_block->m_block_angle_bin_count);
    }

    if (max_bin % m_angle_group_size != 0) {
        max_bin += m_angle_group_size - (max_bin % m_angle_group_size);
    }
}

template <class Element_type>
void Part<Element_type>::compute_y_ax_cscvb_blocked() {
    m_part_detector->set_zero();
    for (Block<Element_type>* block: m_blocks) {
        CSCVB_matrix_block<Element_type>* cscvb = block->m_cscvb_block;

        // 这里要怎么展开比较好呢？
        if (m_angle_group_size == 4) {
            cscvb->template compute_y_ax_tuple_common<4>(block->m_block_image, block->m_block_detector);
            cscvb->template store_y<4>(m_part_detector, m_num_bin, m_start_angle, block->m_start_angle, block->m_block_detector);
        } else if (m_angle_group_size == 8) {
            cscvb->template compute_y_ax_tuple_common<8>(block->m_block_image, block->m_block_detector);
            cscvb->template store_y<8>(m_part_detector, m_num_bin, m_start_angle, block->m_start_angle, block->m_block_detector);
        } else if (m_angle_group_size == 16) {
            cscvb->template compute_y_ax_tuple_common<16>(block->m_block_image, block->m_block_detector);
            cscvb->template store_y<16>(m_part_detector, m_num_bin, m_start_angle, block->m_start_angle, block->m_block_detector);
        } else {
            // TODO: provide a common block kernel, supports arbitary block size, while provuding warning for wrong block sizes
            ASSERT_AND_PRINTF(false, "block kernel not found, angle group = %d\n", m_angle_group_size);
        }
    }
}

template <class Element_type>
void Part<Element_type>::compute_y_ax_coo_blocked() {
    m_part_detector->set_zero();
    for (Block<Element_type>* block: m_blocks) {
        block->m_coo_block->multiply_dense_vector(*block->m_block_image, *block->m_block_detector);
        int y_reduction_dst = m_num_bin * (block->m_start_angle - m_start_angle);

        ASSERT_AND_PRINTF(block->m_block_detector->get_size() == block->m_angle_count * m_num_bin, "%d %d\n",
                            block->m_block_detector->get_size(), block->m_angle_count * m_num_bin);
        for (int i = 0; i < block->m_angle_count * m_num_bin; i++) {
            m_part_detector->at(y_reduction_dst + i) += block->m_block_detector->at(i);
        }
    }
}

template <class Element_type>
void Part<Element_type>::compute_y_ax_csc_blocked() {
    m_part_detector->set_zero();
    for (Block<Element_type>* block: m_blocks) {
        block->m_csc_block->multiply_dense_vector(*block->m_block_image, *block->m_block_detector);
        int y_reduction_dst = m_num_bin * (block->m_start_angle - m_start_angle);

        ASSERT_AND_PRINTF(block->m_block_detector->get_size() == block->m_angle_count * m_num_bin, "%d %d\n",
                            block->m_block_detector->get_size(), block->m_angle_count * m_num_bin);
        for (int i = 0; i < block->m_angle_count * m_num_bin; i++) {
            m_part_detector->at(y_reduction_dst + i) += block->m_block_detector->at(i);
        }
    }
}

template <class Element_type>
void Part<Element_type>::fetch_x(const Image_CT<Element_type>* x) {
    const Element_type* src = &x->at(m_start_x, m_start_y);
    Element_type* dst = &m_part_image->at(0, 0);

    for (int img_y = 0; img_y < m_size_image_y; img_y++) {
        for (int img_x = 0; img_x < m_size_image_x; img_x++) {
            dst[img_x + img_y * m_size_image_x] = src[img_x + img_y * x->m_x];
        }
    }
}

template <class Element_type>
void Part<Element_type>::fetch_y(const Dense_vector<Element_type>* y) {
    Element_type* dst = &m_part_detector->at(0);
    const Element_type* src = &y->at(m_start_angle * m_num_bin);
    for (int angle0 = 0; angle0 < m_angle_count; angle0++) {
        for (int bin = m_y_lhss->at(angle0); bin <= m_y_rhss->at(angle0); bin++) {
            dst[angle0 * m_num_bin + bin] = src[angle0 * m_num_bin + bin];
        }
    }
}

template <class Element_type>
std::vector<CSCV_block_statistics*> Part<Element_type>::get_cscv_statistics_in_blocks() const {
    std::vector<CSCV_block_statistics*> stas;

    for (auto* block : m_blocks) {
        stas.push_back(block->m_cscv_sta);
    }

    return stas;
}
