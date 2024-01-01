#include "system_matrix.hpp"

#include "base/seq_map.hpp"

struct Col_data_struct {
    int row[5];
    float m_value[5];
    // note：想想优化这个函数的写法。
    // 微缩版的 sparse vector
    void add(int index, float value) {
        for (int i = 0; i < 5; i++) {
            if (this->row[i] == index) {
                this->m_value[i] += value;
                break;
            } else if (this->row[i] == -1) {
                this->row[i] = index;
                this->m_value[i] = value;
                break;
            }
        }
    }

    void reset() {
        for (int i = 0; i < 5; i++) {
            row[i] = -1;
            m_value[i] = -100.0;
        }
    }
};

struct Pair_triangle_array {
    int m_node_idx_of_first_triangle[2][3];
    Block_shape m_range_pair_triangle;  // to be replaced

    void template_triangle_init(int dim_block) {
        m_node_idx_of_first_triangle[0][0] = 0;
        m_node_idx_of_first_triangle[0][1] = 1;
        m_node_idx_of_first_triangle[0][2] = dim_block;
        m_node_idx_of_first_triangle[1][0] = dim_block + 1;
        m_node_idx_of_first_triangle[1][1] = dim_block;
        m_node_idx_of_first_triangle[1][2] = 1;
    }

    void init(const Block_shape *range_img_surrounded, int img_size) {
        int i;
        int tri_idx_start[2];
        int tri_idx_end[2];
        template_triangle_init(range_img_surrounded->m_dim_block[1]);
        // 点到三角形的映射法则
        for (i = 0; i <= 1; i++) {
            // 4*4 img, 6*6 imgSurrounded, 5*5 triangle_mesh:
            // idxEnd_mesh: 4,4(5-1), idxEnd_mesh_not_cross_borderOfImgSurrounded: 3,3
            if (range_img_surrounded->m_idx_start[i] < 0)  // 如果超出了左上边界，就不取；不取与边界相关的三角形
                tri_idx_start[i] = 1;
            else
                tri_idx_start[i] = 0;

            // 如果超出了右下边界，就不取；不取与边界相关的三角形
            if (range_img_surrounded->m_idx_end[i] > img_size - 1)
                tri_idx_end[i] = range_img_surrounded->m_dim_block[i] - 3;
            else
                tri_idx_end[i] = range_img_surrounded->m_dim_block[i] - 2;
        }

        m_range_pair_triangle.init(tri_idx_start[0], tri_idx_end[0], tri_idx_start[1], tri_idx_end[1]);
    }
};

void sort_points(const int *index, int *index_sort, const struct Float_point *points) {
    // 检查一下
    // 排序
    int temp_int;
    for (int i = 0; i < 3; i++)
        index_sort[i] = i;

    // 考虑优先级别；冒泡排序
    for (int i = 0; i < 2; i++)
        for (int j = i + 1; j < 3; j++)
            if (points[index[index_sort[i]]].m_x > points[index[index_sort[j]]].m_x) {
                temp_int = index_sort[i];
                index_sort[i] = index_sort[j];
                index_sort[j] = temp_int;
            }
}

int calc_integral_coef_triangles(const struct Float_point *points, const int index[3], const float *bin_list,
           int *bin_range, float len_bin, int num_bin, float (*value)[3]) {
    // 判断跨越哪个bin
    // 排序
    // float check_sum = 0.0f;
    int flag;
    float t1, t2, t1t2;
    int index_sort[3];

    // 初始化
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            value[i][j] = 0;

    // sort index_sort( initialized as {0, 1, 2} ) by points[index[index_sort[i]]].x
    sort_points(index, index_sort, points);
    bin_range[0] = static_cast<int>(ceil((points[index[index_sort[0]]].m_x - bin_list[0] + EPSILON) / len_bin));
    bin_range[1] = static_cast<int>(floor((points[index[index_sort[2]]].m_x - bin_list[0] - EPSILON) / len_bin));

    // CHECK_INT_RANGE(bin_range[0], 0, num_bin - 1);
    // CHECK_INT_RANGE(bin_range[1], 0, num_bin - 1);

    // 改变了的index，算出了
    // 判断相交0-1号点
    // check:临界的情况与公式
    flag = bin_range[1] - bin_range[0];
    switch (flag) {  // if(bin_range[1]-bin_range[0] == 0) // 一条边的情况:分交2条直角边，还是交一直角边一斜边——似乎公式一样
    case 0: {  // 结果存在2个value上
        // flag = 0;
        if (points[index[index_sort[1]]].m_x - bin_list[bin_range[0]] > 0) {
            // 角在左边
            t1 = (bin_list[bin_range[0]] - points[index[index_sort[0]]].m_x) /
                 (points[index[index_sort[1]]].m_x -points[index[index_sort[0]]].m_x);
            t2 = (bin_list[bin_range[0]] - points[index[index_sort[0]]].m_x) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[0]]].m_x);
            t1t2 = t1 * t2 / 6;
            // 直角公式;直角边角同样处理
            value[0][index_sort[1]] = t1t2 * t1;
            value[0][index_sort[2]] = t1t2 * t2;
            value[0][index_sort[0]] = 3.0f * t1t2 - value[0][index_sort[1]] - value[0][index_sort[2]];
            for (int i = 0; i < 3; i++)
                value[1][index_sort[i]] = 1.0f / 6.0f - value[0][index_sort[i]];  // 优化
        } else {
            // 角在右边 //还可以优
            t1 = (points[index[index_sort[2]]].m_x - bin_list[bin_range[0]]) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[0]]].m_x);
            t2 = (points[index[index_sort[2]]].m_x - bin_list[bin_range[0]]) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[1]]].m_x);
            t1t2 = t1 * t2 / 6;
            value[1][index_sort[0]] = t1t2 * t1;
            value[1][index_sort[1]] = t1t2 * t2;
            value[1][index_sort[2]] = 3.0f * t1t2 - value[1][index_sort[0]] - value[1][index_sort[1]];
            for (int i = 0; i < 3; i++)
                value[0][index_sort[i]] = 1.0f / 6.0f - value[1][index_sort[i]];  // 优化
        }
        break;
    }
    case 1: {
        // if(bin_range[1]-bin_range[0] == 1)// ==1的情况，与两条边相交的情况
            // flag = 1;
            t1 = (bin_list[bin_range[0]] - points[index[index_sort[0]]].m_x) /
                 (points[index[index_sort[1]]].m_x - points[index[index_sort[0]]].m_x);
            t2 = (bin_list[bin_range[0]] - points[index[index_sort[0]]].m_x) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[0]]].m_x);
            t1t2 = t1 * t2 / 6;
            value[0][index_sort[1]] = t1t2 * t1;
            value[0][index_sort[2]] = t1t2 * t2;
            value[0][index_sort[0]] = 3.0f * t1t2 - value[0][index_sort[1]] - value[0][index_sort[2]];

            t1 = (points[index[index_sort[2]]].m_x - bin_list[bin_range[1]]) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[0]]].m_x);
            t2 = (points[index[index_sort[2]]].m_x - bin_list[bin_range[1]]) /
                 (points[index[index_sort[2]]].m_x - points[index[index_sort[1]]].m_x);
            t1t2 = t1 * t2 / 6;
            value[2][index_sort[0]] = t1t2 * t1;
            value[2][index_sort[1]] = t1t2 * t2;
            value[2][index_sort[2]] = 3.0f * t1t2 - value[2][index_sort[0]] - value[2][index_sort[1]];

            for (int i = 0; i < 3; i++)
                value[1][index_sort[i]] = 1.0f / 6.0f - value[0][index_sort[i]] - value[2][index_sort[i]];
            break;
        }
    case -1: {
        for (int i = 0; i < 3; i++)
            value[0][i] = 1.0f / 6.0f;
        break;
    }
    default: {
        printf("Warning!!\n");  // 块状切分，行切15块的时候出错。
    }
    }

    // 输出value和bin_range；
    return flag;
}

struct System_matrix_buffer {
    
    float *m_array_bin_centre_coord;
    Float_point *m_points, *m_points_rotate;
    Col_data_struct *m_tmp_col_data;

    System_matrix_buffer(const Img_param& img_param) {
        m_array_bin_centre_coord = reinterpret_cast<float*>(malloc(sizeof(float) * (img_param.m_num_bin + 1)));
        float coord_detector_centre = 0.5f * (img_param.m_num_bin + 1);  // 1号bin和n号bin的中心
        for (int i = 0; i < img_param.m_num_bin + 1; i++)
            m_array_bin_centre_coord[i] = 0.5f + i - coord_detector_centre;  // 第i个bin的归一化的中心坐标。

        int points_dim_row = img_param.m_global_img_size + 2 * Img_param::m_half_num_rows_out_border;
        int points_dim_col = img_param.m_global_img_size + 2 * Img_param::m_half_num_cols_out_border;

        m_points = new Float_point[points_dim_row * points_dim_col];
        m_points_rotate = new Float_point[points_dim_row * points_dim_col];
        m_tmp_col_data = new Col_data_struct[img_param.m_global_img_size * img_param.m_global_img_size];
    }

    ~System_matrix_buffer() {
        free(m_array_bin_centre_coord);
        delete[] m_points_rotate;
        delete[] m_points;
        delete[] m_tmp_col_data;
    }
};

static thread_local Seq_map<Img_param, System_matrix_buffer*> g_sml_gen_buffer;

void generate_sub_system_matrix_linear_inner(Img_param img_param, Range_nd<2> img_range, Range_1d angle_range,
                                             void *fill_target, Matrix_fill_func fill_func) {
    // static __thread Seq_map<Img_param, System_matrix_buffer*> g_sml_gen_buffer;

    // get buffers
    System_matrix_buffer* buffers;
    if (g_sml_gen_buffer.count(img_param) == 0) {
        buffers = new System_matrix_buffer(img_param);
        g_sml_gen_buffer.insert(img_param, buffers);
        // PRINTF("creating new buffer\n");
    } else {
        buffers = g_sml_gen_buffer.at(img_param);
    }

    img_range[0].m_start += img_param.m_global_img_x_start;
    img_range[1].m_start += img_param.m_global_img_y_start;

    Block_shape range_img_block;
    range_img_block.init(img_range[1].m_start,
                         img_range[1].m_start + img_range[1].m_size - 1,
                         img_range[0].m_start,
                         img_range[0].m_start + img_range[0].m_size - 1);

    Block_shape range_img_block_surrounded;
    range_img_block_surrounded.init(range_img_block.m_idx_start[0] - Img_param::m_half_num_rows_out_border,
                                    range_img_block.m_idx_end[0] + Img_param::m_half_num_rows_out_border,
                                    range_img_block.m_idx_start[1] - Img_param::m_half_num_cols_out_border,
                                    range_img_block.m_idx_end[1] + Img_param::m_half_num_cols_out_border);

    const int local_num_points = range_img_block_surrounded.m_dim_block[0] * range_img_block_surrounded.m_dim_block[1];

    Pair_triangle_array triangle_mesh;
    triangle_mesh.init(&range_img_block_surrounded, img_param.m_global_img_size);

    int matrix_row_size = angle_range.m_size * img_param.m_num_bin;
    int matrix_col_size = img_range[1].m_size * img_range[0].m_size;
    int max_nnz_per_angle = matrix_col_size * 4;

    float *array_bin_centre_coord = buffers->m_array_bin_centre_coord;

    Float_point *points = buffers->m_points,
                *points_rotate = buffers->m_points_rotate;

    Col_data_struct *tmp_col_data = buffers->m_tmp_col_data;

    int64_t item_total = 0;
    int item_local;
    int64_t item = 0;

    int idx = 0;
    for (int i = 0; i < range_img_block_surrounded.m_dim_block[0]; i++) {
        for (int j = 0; j < range_img_block_surrounded.m_dim_block[1]; j++) {
            points[idx].m_x = static_cast<float>(j + range_img_block_surrounded.m_idx_start[1]);
            points[idx].m_y = static_cast<float>(i + range_img_block_surrounded.m_idx_start[0]);
            points[idx].m_x -= img_param.m_rotate_centre[0];
            points[idx].m_y -= img_param.m_rotate_centre[1];
            idx++;
        }
    }

    for (int angle_id = angle_range.m_start; angle_id < angle_range.m_start + angle_range.m_size; angle_id++) {
        float theta = (img_param.m_start_angle + img_param.m_delta_angle * angle_id) / 180.0f * PI;
        Float_point::rotate(points_rotate, points, theta, /*img_size*/ local_num_points);
        int sub_row_offset = (angle_id - angle_range.m_start) * img_param.m_num_bin;

        for (int i = 0; i < matrix_col_size; i++)
            tmp_col_data[i].reset();

        for (int k = triangle_mesh.m_range_pair_triangle.m_idx_start[0];
                 k <= triangle_mesh.m_range_pair_triangle.m_idx_end[0]; k++) {
            for (int l = triangle_mesh.m_range_pair_triangle.m_idx_start[1];
                     l <= triangle_mesh.m_range_pair_triangle.m_idx_end[1]; l++) {
                for (int t = 0; t < 2; t++) {
                    int node_idx_of_loop_triangle[3];
                    for (int m = 0; m < 3; m++)
                        node_idx_of_loop_triangle[m] = triangle_mesh.m_node_idx_of_first_triangle[t][m] +
                                                       k * range_img_block_surrounded.m_dim_block[1] + l;

                    int range_bin_of_triangle[2];
                    float tmp_value_of_triangle[3][3];

                    int flag = calc_integral_coef_triangles(points_rotate, node_idx_of_loop_triangle, array_bin_centre_coord,
                                                            range_bin_of_triangle, Img_param::m_len_bin, img_param.m_num_bin,
                                                            tmp_value_of_triangle);

                    for (int m = 0; m < 3; m++) {
                        int node2DIdxInSurroundedImg[2];
                        node2DIdxInSurroundedImg[0] = node_idx_of_loop_triangle[m] / range_img_block_surrounded.m_dim_block[1];
                        node2DIdxInSurroundedImg[1] = node_idx_of_loop_triangle[m] % range_img_block_surrounded.m_dim_block[1];

                        if (range_img_block_surrounded.check_if_point_inside(node2DIdxInSurroundedImg) == 0) {
                            continue;
                        } else {
                            int node2DIdxInImg[2];
                            int tmp_col_index;
                            int tmp_row_index;
                            node2DIdxInImg[0] = node2DIdxInSurroundedImg[0] - Img_param::m_half_num_rows_out_border;
                            node2DIdxInImg[1] = node2DIdxInSurroundedImg[1] - Img_param::m_half_num_cols_out_border;
                            tmp_col_index = node2DIdxInImg[0] * range_img_block.m_dim_block[1] + node2DIdxInImg[1];

                            assert(tmp_col_index < matrix_col_size);

                            tmp_row_index = sub_row_offset + range_bin_of_triangle[0] - 1;
                            if (tmp_row_index >= 0 && tmp_row_index < matrix_row_size)
                                tmp_col_data[tmp_col_index].add(tmp_row_index,
                                                                img_param.m_pixel_ratio * tmp_value_of_triangle[0][m]);

                            tmp_row_index = sub_row_offset + range_bin_of_triangle[0];
                            if (flag >= 0 && tmp_row_index >= 0 && tmp_row_index < matrix_row_size)
                                tmp_col_data[tmp_col_index].add(tmp_row_index,
                                                                img_param.m_pixel_ratio * tmp_value_of_triangle[1][m]);

                            tmp_row_index = sub_row_offset + range_bin_of_triangle[1];
                            if (flag == 1 && tmp_row_index >= 0 && tmp_row_index < matrix_row_size)
                                tmp_col_data[tmp_col_index].add(tmp_row_index,
                                                                img_param.m_pixel_ratio * tmp_value_of_triangle[2][m]);
                        }
                    }
                }
            }
        }

        item_local = 0;
        double checksum = 0.0;

        for (int i = 0; i < matrix_col_size; i++) {
            for (int j = 0; j < 5; j++) {
                if (tmp_col_data[i].row[j] == -1)
                    break;
                int row = tmp_col_data[i].row[j];
                int col = i;
                float value = tmp_col_data[i].m_value[j];
                checksum += static_cast<double>(value);
                item_local++;

                fill_func(fill_target, row, col, value);
            }
        }

        if (item_local > max_nnz_per_angle)
            assert(false);
    }
}

static inline void rotate_(double rotate_pixel[2], double pixel[2], double theta) {
    rotate_pixel[0] = cos(theta)*pixel[0] - sin(theta)*pixel[1];
    rotate_pixel[1] = sin(theta)*pixel[0] + cos(theta)*pixel[1];
}

static inline int IDConvert2DTo1D(int ID[2], int dim[2]) {
    return ID[0] * dim[1] + ID[1];
}

constexpr int DIM_ROW = 0, DIM_COL = 1;

void generate_sub_system_matrix_constant_inner(Img_param img_param, Range_nd<2> img_range, Range_1d angle_range,
                                             void *fill_target, Matrix_fill_func fill_func) {
    // static __thread Seq_map<Img_param, System_matrix_buffer*> g_sml_gen_buffer;

    // get buffers
    System_matrix_buffer* buffers;
    if (g_sml_gen_buffer.count(img_param) == 0) {
        buffers = new System_matrix_buffer(img_param);
        g_sml_gen_buffer.insert(img_param, buffers);
    } else {
        buffers = g_sml_gen_buffer.at(img_param);
    }

    img_range[0].m_start += img_param.m_global_img_x_start;
    img_range[1].m_start += img_param.m_global_img_y_start;

    const int img_size = img_param.m_global_img_size;
    const int num_bin = img_param.m_num_bin;

    float len_bin = img_param.m_size_pixel;
    float pixelRatio = img_param.m_pixel_ratio;

    double theta;
    double rotate_centre[2], shift_rotate_centre[2];
    double pixel[2], pixel_rotate[2];
    double subpixel[4][2], rotate_subpixel[4][2];
    double operate_pixel[2];
    double bin_centre;
    int i, j;

    double ratio1;
    double y_bin;
    int index_y_bin;

    long long item = 0;
    long long item_total = 0;
    int item_local = 0;

    int local_numPixels = img_range[0].m_size * img_range[1].m_size;
    int nnzSupremum_SingleAngle = 4 * local_numPixels;

    float delta_angle = img_param.m_delta_angle;
    float start_angle = img_param.m_start_angle;

    rotate_centre[0] = 0.5 * (img_size - 1);
    rotate_centre[1] = 0.5 * (img_size - 1);

    bin_centre = 0.5 * (num_bin + 1);

    Col_data_struct *tmp_col_data = buffers->m_tmp_col_data;

    float *arrayBinCentreCoord = buffers->m_array_bin_centre_coord;

    shift_rotate_centre[0] = -rotate_centre[0];
    shift_rotate_centre[1] = -rotate_centre[1];

    subpixel[0][0] = -0.25;
    subpixel[0][1] = -0.25;
    subpixel[1][0] = 0.25;
    subpixel[1][1] = -0.25;
    subpixel[2][0] = -0.25;
    subpixel[2][1] = 0.25;
    subpixel[3][0] = 0.25;
    subpixel[3][1] = 0.25;

    int local_IdxImgStart[2];
    local_IdxImgStart[DIM_ROW] = img_range[1].m_start;
    local_IdxImgStart[DIM_COL] = img_range[0].m_start;

    int matrix_row_size = angle_range.m_size * img_param.m_num_bin;
    int matrix_col_size = img_range[1].m_size * img_range[0].m_size;
    int angleNum = angle_range.m_size;

    int max_nnz_per_angle = matrix_col_size * 4;

    for (int angle_id = angle_range.m_start; angle_id < angle_range.m_start + angle_range.m_size; angle_id++) {
        theta = (start_angle + delta_angle * angle_id) / 180.0f * PI;

        for (int i = 0; i < matrix_col_size; i++)
            tmp_col_data[i].reset();

        for (int k = 0; k < 4; k++)
            rotate_(rotate_subpixel[k], subpixel[k], theta);

        for (int y = 0; y < img_range[1].m_size; y++) {
            for (int x = 0; x < img_range[0].m_size; x++) {
                int tmpColIdx = x + y * img_range[0].m_size;

                pixel[0] = (double)(x + local_IdxImgStart[DIM_COL]) + shift_rotate_centre[0];
                pixel[1] = (double)(y + local_IdxImgStart[DIM_ROW]) + shift_rotate_centre[1];

                rotate_(pixel_rotate, pixel, theta);
                int rowOffset = (angle_id - angle_range.m_start) * num_bin;

                for (int k = 0; k < 4; k++) {
                    operate_pixel[0] = pixel_rotate[0] + rotate_subpixel[k][0];
                    operate_pixel[1] = pixel_rotate[1] + rotate_subpixel[k][1];

                    y_bin = operate_pixel[0] + bin_centre;
                    index_y_bin = (int)(y_bin);
                    ratio1 = y_bin - floor(y_bin);

                    int tmpRowIdx = index_y_bin + rowOffset;
                    if (tmpRowIdx >= 0 && tmpRowIdx < matrix_row_size)
                        tmp_col_data[tmpColIdx].add(tmpRowIdx, pixelRatio * 0.25 * ratio1);
                    tmpRowIdx = index_y_bin - 1 + rowOffset;
                    if (tmpRowIdx >= 0 && tmpRowIdx < matrix_row_size)
                        tmp_col_data[tmpColIdx].add(tmpRowIdx, pixelRatio * 0.25 * (1 - ratio1));
                }
            }
        }

        double checksum = 0.0;

        item_local = 0;
        for (int i = 0; i < matrix_col_size; i++) {
            for (int j = 0; j < 5; j++) {
                if (tmp_col_data[i].row[j] == -1)
                    break;
                item_local++;
                int row = tmp_col_data[i].row[j];
                int col = i;
                float value = tmp_col_data[i].m_value[j];
                checksum += static_cast<double>(value);

                fill_func(fill_target, row, col, value);
            }
        }

        ASSERT_AND_PRINTF(item_local <= max_nnz_per_angle, "%d > %d\n", item_local, max_nnz_per_angle);
    }
}

void free_system_matrix_gen_tmp_data() {
    auto l = g_sml_gen_buffer.get_list();

    for (auto p : l) {
        delete p.second;
    }
}
