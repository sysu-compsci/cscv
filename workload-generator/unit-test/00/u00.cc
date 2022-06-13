#include "base/result_table.hpp"
#include "ct/system_matrix.hpp"

#include <iostream>

using namespace std;

template <class Element_type>
void test_gen_and_compute(const Img_param& img_param) {
    System_matrix_generator mtx_generator(img_param);

    Range_nd<2> img_range;
    img_range[0] = Range_1d(0, img_param.m_img_size);
    img_range[1] = Range_1d(0, img_param.m_img_size);
    Range_1d angle_range = Range_1d(0, img_param.m_num_angle);  // <start, size>

    // COO_matrix_buffer<Element_type>* coo_A = mtx_generator.generate_system_matrix<Element_type>(img_range, angle_range);
    COO_matrix_buffer<Element_type>* coo_A = mtx_generator.generate_system_matrix_constant<Element_type>(img_range, angle_range);
    PRINTF("generation finished, %s\n", coo_A->get_summary_string().c_str());

    coo_A->write_to_txt(strprintf("glb_img_%d_xs_%d_ys_%d_img_%d_bin_%d_sa_%f_da_%f_na_%d.txt",
                        img_param.m_global_img_size, img_param.m_global_img_x_start, img_param.m_global_img_y_start,
                        img_param.m_img_size, img_param.m_num_bin, img_param.m_start_angle, img_param.m_delta_angle, img_param.m_num_angle));

    return;

    CSR_matrix<Element_type>* csr_A = coo_A->convert_to_csr_matrix();
    CSC_matrix<Element_type>* csc_A = coo_A->convert_to_csc_matrix();

    sparse_matrix_t* csr_mkl_A = csr_A->convert_to_mkl_matrix();
    sparse_matrix_t* csc_mkl_A = csc_A->convert_to_mkl_matrix();

    Dense_vector<Element_type> x(img_param.m_img_size * img_param.m_img_size);
    for (int i = 0; i < img_param.m_img_size * img_param.m_img_size; i++)
        x.at(i) = 1.0f;

    Dense_vector<Element_type> y_coo(img_param.m_num_angle * img_param.m_num_bin);
    Dense_vector<Element_type> y_csr(img_param.m_num_angle * img_param.m_num_bin);
    Dense_vector<Element_type> y_csc(img_param.m_num_angle * img_param.m_num_bin);
    Dense_vector<Element_type> y_csr_mkl(img_param.m_num_angle * img_param.m_num_bin);
    Dense_vector<Element_type> y_csc_mkl(img_param.m_num_angle * img_param.m_num_bin);

    coo_A->multiply_dense_vector(x, y_coo);
    csr_A->multiply_dense_vector(x, y_csr);
    csc_A->multiply_dense_vector(x, y_csc);

    matrix_descr mtx_descr;
    mtx_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t mkl_ret;

    if constexpr (std::is_same<Element_type, float>::value) {
        mkl_ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *csr_mkl_A, mtx_descr,
                                  &(x.at(0)), 0.0, &(y_csr_mkl.at(0)));
        check_mkl_sparse_ret(mkl_ret);
        mkl_ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *csc_mkl_A, mtx_descr,
                                  &(x.at(0)), 0.0, &(y_csc_mkl.at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else if constexpr (std::is_same<Element_type, double>::value) {
        mkl_ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *csr_mkl_A, mtx_descr,
                                  &(x.at(0)), 0.0, &(y_csr_mkl.at(0)));
        check_mkl_sparse_ret(mkl_ret);
        mkl_ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *csc_mkl_A, mtx_descr,
                                  &(x.at(0)), 0.0, &(y_csc_mkl.at(0)));
        check_mkl_sparse_ret(mkl_ret);
    } else {
        assert(false);
    }

    Result_table<Element_type> rt;
    rt.set_label(0, "coo");
    rt.set_label(1, "csr");
    rt.set_label(2, "csc");
    rt.set_label(3, "csr mkl");
    rt.set_label(4, "csc mkl");

    rt.collect_data_arr(0, img_param.m_num_angle * img_param.m_num_bin, &y_coo.at(0));
    rt.collect_data_arr(1, img_param.m_num_angle * img_param.m_num_bin, &y_csr.at(0));
    rt.collect_data_arr(2, img_param.m_num_angle * img_param.m_num_bin, &y_csc.at(0));
    rt.collect_data_arr(3, img_param.m_num_angle * img_param.m_num_bin, &y_csr_mkl.at(0));
    rt.collect_data_arr(4, img_param.m_num_angle * img_param.m_num_bin, &y_csc_mkl.at(0));

    printf("%s\n", rt.get_summary_diff_string("y diff").c_str());
}

int main() {
    int global_img_x_start = 0;
    int global_img_y_start = 0;
    int global_img_size = 128;
    int img_size = 128;
    int num_angle = 64;
    int num_bin = 200;
    double delta_angle = 2;

    printf("[CSCV workload generator] Generating CT-SART matrix\n");
    std::cout << "global_img_x_start: " << global_img_x_start << std::endl;
    std::cout << "global_img_y_start: " << global_img_y_start << std::endl;
    std::cout << "global_img_size: " << global_img_size << std::endl;
    std::cout << "img_size: " << img_size << std::endl;
    std::cout << "num_angle: " << num_angle << std::endl;
    std::cout << "num_bin: " << num_bin << std::endl;
    std::cout << "delta_angle: " << delta_angle << std::endl;

    Img_param img_param(img_size, 0.0, delta_angle, num_angle, num_bin, 1.0);
    img_param.set_global_img_coord(global_img_x_start, global_img_y_start);
    img_param.set_global_img_size(global_img_size);

    test_gen_and_compute<float>(img_param);
    // test_gen_and_compute<double>(img_param);

    return 0;
}
