#include "partition.hpp"

using namespace std;

Partitioner_cscv::Partitioner_cscv(Img_param img_param, int angle_group, int x_group, int y_group, int angle_part, int x_part, int y_part) {
    ASSERT_AND_PRINTF(img_param.m_img_size % x_group == 0, "undivided img x %d %d", img_param.m_img_size, x_group);
    ASSERT_AND_PRINTF(img_param.m_img_size % y_group == 0, "undivided img y %d %d", img_param.m_img_size, y_group);

    // ASSERT_AND_PRINTF(img_param.m_num_angle % angle_group == 0, "undivided angle %d %d", img_param.m_num_angle, angle_group);

    m_img_param = img_param;

    m_x_group_size = x_group;
    m_y_group_size = y_group;
    m_angle_group_size = angle_group;

    m_x_part = x_part;
    m_y_part = y_part;
    m_angle_part = angle_part;

    // get block count
    m_block_angle = div_and_ceil(img_param.m_num_angle, angle_group);

    m_img_x_block_count = img_param.m_img_size / x_group;
    m_img_y_block_count = img_param.m_img_size / y_group;

    // get partition on angle and block: how many part in each dimension
    for (int i = 0; i < m_x_part + 1; i++) {
        m_img_x_block_id_offsets.push_back(BLOCK_LOW(i, m_x_part, m_img_x_block_count));
    }

    for (int i = 0; i < m_y_part + 1; i++) {
        m_img_y_block_id_offsets.push_back(BLOCK_LOW(i, m_y_part, m_img_y_block_count));
    }

    for (int i = 0 ; i < m_angle_part + 1; i++) {
        m_angle_block_id_offsets.push_back(BLOCK_LOW(i, m_angle_part, m_block_angle));
    }

    m_order_label_map[Block_order::ANGLE_X_Y] = "ANGLE_X_Y";
    m_order_label_map[Block_order::ANGLE_Y_X] = "ANGLE_Y_X";
    m_order_label_map[Block_order::X_ANGLE_Y] = "X_ANGLE_Y";
    m_order_label_map[Block_order::X_Y_ANGLE] = "X_Y_ANGLE";
    m_order_label_map[Block_order::Y_ANGLE_X] = "Y_ANGLE_X";
    m_order_label_map[Block_order::Y_X_ANGLE] = "Y_X_ANGLE";

    set_block_order(Block_order::ANGLE_X_Y);
    set_pxg_size(2);
}

string Partitioner_cscv::get_summary_string() const {
    stringstream ss;
    ss << "{";

    ss << strprintf(" { img info: {img_size: %d, angle_count: %d, delta angle = %f\n} }", m_img_param.m_img_size, m_img_param.m_num_angle, m_img_param.m_delta_angle);
    ss << strprintf(" { group size: {x: %d, y: %d, angle: %d} }, ", m_x_group_size, m_y_group_size, m_angle_group_size);
    ss << strprintf(" { part dim: {x: %d, y: %d, angle: %d} }, ", m_x_part, m_y_part, m_angle_part);
    ss << strprintf(" { x part offsets: %s, y part offsets: %s, angle part offsets: %s}",
                     vector_to_string(m_img_x_block_id_offsets).c_str(), vector_to_string(m_img_y_block_id_offsets).c_str(), vector_to_string(m_angle_block_id_offsets).c_str());

    ss << "}";
    return ss.str();
}

void Partitioner_cscv::set_block_order(Block_order order) {
    m_block_order = order;
    PRINTF("block order set to %s\n", m_order_label_map.at(order).c_str());
}

void Partitioner_cscv::set_pxg_size(int pxg_size) {
    m_pxg_size = pxg_size;
    PRINTF("pxg size set to %d\n", pxg_size);
}

Partition_result_cscv Partitioner_cscv::get_partition_result() {
    Partition_result_cscv partition_result;
    partition_result.m_x_group_size = m_x_group_size;
    partition_result.m_y_group_size = m_y_group_size;
    partition_result.m_angle_group_size = m_angle_group_size;
    partition_result.m_pxg_size = m_pxg_size;
    partition_result.m_img_x_block_count = m_img_x_block_count;
    partition_result.m_img_y_block_count = m_img_y_block_count;
    partition_result.m_block_angle = m_block_angle;
    partition_result.m_angle_part = m_angle_part;
    partition_result.m_x_part = m_x_part;
    partition_result.m_y_part = m_y_part;
    partition_result.m_img_x_block_id_offsets = m_img_x_block_id_offsets;
    partition_result.m_img_y_block_id_offsets = m_img_y_block_id_offsets;
    partition_result.m_angle_block_id_offsets = m_angle_block_id_offsets;

    // Data_holder<Element_type>* ret = new Data_holder<Element_type>(m_img_param, build_parts<Element_type>(), partition_result, m_comp_cfg);
    // PRINTF("data holder generated, non-empty part count = %d\n", ret->m_part_count);

    return partition_result;
}
