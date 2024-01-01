#include "cscv/soa.tcc"

template <uint8_t t_vec_angle, uint8_t t_px_group>
static void fake_func_inner() {
    printf("%p\n", &tea_blocks_compute_y_ax<float, t_vec_angle, t_px_group, true>);
}

template <uint8_t t_vec_angle>
static void fake_func_pxg_expand() {
    fake_func_inner<t_vec_angle, 1>();
    fake_func_inner<t_vec_angle, 2>();
    fake_func_inner<t_vec_angle, 4>();
    fake_func_inner<t_vec_angle, 8>();
    fake_func_inner<t_vec_angle, 16>();
    fake_func_inner<t_vec_angle, 32>();
}

void fake_func_cat_float() {
    fake_func_pxg_expand<4>();
    fake_func_pxg_expand<8>();
    fake_func_pxg_expand<16>();
}
