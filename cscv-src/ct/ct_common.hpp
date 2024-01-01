#pragma once

#include <assert.h>
#include <stdio.h>

#include <cmath>
#include <string>

#include "base/basic_definition.hpp"

#define EPSILON 1e-8
#define PI 3.1415926535898f

struct Img_param {
    // const info
    static constexpr float m_len_bin = 1;
    static constexpr int m_num_rows_out_border = 2;
    static constexpr int m_half_num_rows_out_border = m_num_rows_out_border >> 1;
    static constexpr int m_num_cols_out_border = 2;
    static constexpr int m_half_num_cols_out_border = m_num_cols_out_border >> 1;

    // global img param, just related to the system matrix generation
    int m_global_img_size = -1;
    int m_global_img_x_start = 0, m_global_img_y_start = 0;

    // local img param configuration
    int m_img_size;  // 正方形图像的宽度
    int m_num_angle;  // 探测角度数量
    int m_num_bin;  // 探测器数量
    float m_delta_angle;  // 180 / num_angles？
    float m_start_angle;  // 精度可配置
    float m_size_pixel;  // 每个像素的大小是 bin 距离的多少倍，一般为 1, 总感觉好像有问题

    // calculated
    float m_rotate_centre[2];
    float m_pixel_ratio;  // m_size_pixel 的平方

    std::string debug_string() const {
        char c[200];
        snprintf(c, sizeof(c), "m_size_img = %d, m_num_angle = %d, m_num_bins = %d, "
                          "m_delta_angle = %f, m_start_angle = %f, m_size_pixel = %f",
                 m_img_size, m_num_angle, m_num_bin, m_delta_angle, m_start_angle, m_size_pixel);
        return std::string(c);
    }

    std::string filename_string() const {
        return strprintf("gs_%d_gx_%d_gy_%d_s_%d_a_%d_bin_%d_da_%f_sa_%f",
                          m_global_img_size, m_global_img_x_start, m_global_img_y_start,
                          m_img_size, m_num_angle, m_num_bin, m_delta_angle, m_start_angle);
    }

    float get_y_offset_from_x_coord(Point_nd<2> x_coord, int angle_id) const {
        float actual_angle = m_start_angle + m_delta_angle * angle_id;
        float actual_angle_radian = actual_angle * PI / 180.0;

        float center_y_offset = m_num_bin / 2.0;

        int dim0_offset = x_coord[0] - m_rotate_centre[0];
        int dim1_offset = x_coord[1] - m_rotate_centre[1];

        return cos(actual_angle_radian) * dim0_offset - sin(actual_angle_radian) * dim1_offset + center_y_offset;
    }

    void set_image_size(int size) {
        if (m_global_img_size <= 0)
            set_global_img_size(size);
        m_img_size = size;
    }

    void set_global_img_size(int size) {
        if (size <= 0)
            return;
        m_global_img_size = size;
        m_rotate_centre[0] = m_rotate_centre[1] = 0.5 * (size - 1);
    }

    void set_global_img_coord(int x, int y) {
        m_global_img_x_start = x;
        m_global_img_y_start = y;
    }

    void set_num_angle(int num_angle) {
        m_num_angle = num_angle;
    }

    void set_num_bin(int num_bin) {
        m_num_bin = num_bin;
    }

    void set_delta_angle(float delta_angle) {
        m_delta_angle = delta_angle;
    }

    void set_start_angle(float start_angle) {
        m_start_angle = start_angle;
    }

    void set_size_pixel(float size_pixel) {
        m_size_pixel = size_pixel;
        m_pixel_ratio = size_pixel * size_pixel;
    }

    Img_param() {}

    Img_param(int img_size, float start_angle, float delta_angle, int num_angle, int num_bin, float size_pixel) {
        set_start_angle(start_angle);
        set_delta_angle(delta_angle);
        set_image_size(img_size);
        set_size_pixel(size_pixel);
        set_num_bin(num_bin);
        set_num_angle(num_angle);
    }
};

static inline bool operator==(const Img_param& l, const Img_param& r) {
    if (l.m_img_size != r.m_img_size)
        return false;
    if (l.m_num_angle != r.m_num_angle)
        return false;
    if (l.m_num_bin != r.m_num_bin)
        return false;
    if (l.m_delta_angle != r.m_delta_angle)
        return false;
    if (l.m_start_angle != r.m_start_angle)
        return false;
    if (l.m_size_pixel != r.m_size_pixel)
        return false;
    if (l.m_rotate_centre[0] != r.m_rotate_centre[0])
        return false;
    if (l.m_rotate_centre[1] != r.m_rotate_centre[1])
        return false;
    if (l.m_pixel_ratio != r.m_pixel_ratio)
        return false;
    if (l.m_global_img_size != r.m_global_img_size)
        return false;
    if (l.m_global_img_x_start != r.m_global_img_x_start)
        return false;
    if (l.m_global_img_y_start != r.m_global_img_y_start)
        return false;

    return true;
}

struct Block_shape {
    int m_idx_start[2];
    int m_idx_end[2];
    int m_dim_block[2];

    void init(int _idxRowStart, int _idxRowEnd, int _idxColStart, int _idxColEnd) {
        m_idx_start[0] = _idxRowStart;
        m_idx_start[1] = _idxColStart;
        m_idx_end[0] = _idxRowEnd;
        m_idx_end[1] = _idxColEnd;
        m_dim_block[0] = _idxRowEnd - _idxRowStart + 1;
        m_dim_block[1] = _idxColEnd - _idxColStart + 1;
    }

    int check_if_point_inside(int idx[2]) {
        int flag = 1;

        for (int i = 0; i < 2; i++)
            flag = flag && (idx[i] > 0) && (idx[i] < m_dim_block[i] - 1);

        return flag;
    }
};

struct Float_point {
    float m_x, m_y;

    static void rotate(Float_point *rotate_pixel, const Float_point *pixel, float theta, int length) {
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);
        for (int i = 0; i < length; i++) {
            rotate_pixel[i].m_x = cosTheta * pixel[i].m_x - sinTheta * pixel[i].m_y;
            rotate_pixel[i].m_y = sinTheta * pixel[i].m_x + cosTheta * pixel[i].m_y;
        }
    }
};
