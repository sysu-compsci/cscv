#pragma once

#include <mkl.h>

#include "arch/omp_thread_pool.hpp"
#include "arch/pthread_timer.hpp"
#include "base/basic_definition.hpp"
#include "base/result_table.hpp"
#include "cscv/cscv.hpp"
#include "cscv/soa.hpp"
#include "ct/ct_common.hpp"
#include "ct/ct_image.hpp"
#include "ct/system_matrix.hpp"
#include "data/data_container.hpp"

#include <chrono>

class CSCV_block_statistics;

template <class Element_type>
class CSCV_block_mem_pool;

template <class Element_type>
class System_matrix_converter_cscv;

/**
 * A block containes pixels in the image. These images has the same grouping policies of bins in specific angels.
 **/
template <class Element_type>
struct Block {
    // block parameters below are consistent in a specific configuration
    int m_size_image_x, m_size_image_y;
    int m_angle_count, m_angle_group_size;

    // group sizes
    int m_pxg_size;

    // block position
    int m_start_x, m_start_y;
    int m_start_angle;

    void print_position();

    CSCVB_matrix_block<Element_type>* m_cscvb_block = nullptr;
    CSC_matrix<Element_type>* m_csc_block = nullptr;
    CSR_matrix<Element_type>* m_csr_block = nullptr;
    COO_matrix_buffer<Element_type>* m_coo_block = nullptr;

    void store_x();

    void transpose_y();
    void transpose_y_reverse();

    CSCV_block_statistics* m_cscv_sta = nullptr;

    ~Block();

    template <int t_vec_angle, int t_vec_bin>
    void compute_y_ax();
};

/**
 * A part contains multiple number of blocks.
 **/
template <class Element_type>
struct Part {
    // int m_part_offset;  // != part id; the off set of this part in the array that contains non-empty parts

    // part range
    int m_size_image_x, m_size_image_y;
    int m_angle_count;
    // int m_effective_angle_count;
    int m_num_bin;

    // blocks info
    int m_block_x, m_block_y, m_block_angle;  // dims
    int m_x_group_size, m_y_group_size, m_angle_group_size;
    int m_pxg_size;

    // part position
    int m_start_x, m_start_y;
    int m_start_angle;

    Computation_config m_comp_cfg;

    // global copy
    Img_param m_img_param;
    Dense_vector<Element_type>* m_y_tmp;
    Image_CT<Element_type>* m_x_input;

    // part buffer
    Image_CT<Element_type>* m_part_image = nullptr;
    Dense_vector<Element_type>* m_part_detector = nullptr;

    std::vector<Block<Element_type>*> m_blocks;

    CSCV_block_mem_pool<Element_type>* m_cscv_mem_pool = nullptr;

    CSC_matrix<Element_type>* m_csc_part = nullptr;  // debug
    CSR_matrix<Element_type>* m_csr_part = nullptr;  // debug
    COO_matrix_buffer<Element_type>* m_coo_part = nullptr;  // For debug

    Tea_soa_context<Element_type>* m_tea_soa_context = nullptr;

    Dense_vector<int>* m_y_lhss = nullptr;
    Dense_vector<int>* m_y_rhss = nullptr;

    Ring_fake_barrier* m_y_reduction_barrier;

    ~Part();

    bool is_empty() const;

    size_t get_blocks_coo_nnz() const;

    void build_cscvb_yt_buffer();

    void compute_y_ax_cscvb_blocked();
    void compute_y_ax_coo_blocked();
    void compute_y_ax_csc_blocked();

    void compute_y_ax_csc_by_part();
    void compute_y_ax_coo_by_part();

    void fetch_x(const Image_CT<Element_type>* x);

    void fetch_y(const Dense_vector<Element_type>* y);

    std::vector<CSCV_block_statistics*> get_cscv_statistics_in_blocks() const;

    void build_tea_soa();
};

struct Partition_result_cscv {
    int m_x_group_size, m_y_group_size, m_angle_group_size;  // group size
    int m_pxg_size;
    int m_img_x_block_count, m_img_y_block_count, m_block_angle;  // full block count on each dimension

    int m_angle_part, m_x_part, m_y_part;  // part count, notice that the full block count is their product
    std::vector<int> m_img_x_block_id_offsets, m_img_y_block_id_offsets, m_angle_block_id_offsets;  // offsets of parts
};


/**
 * This class is implemented to generate the partition oriented for CSCV computation.
 * Not related with the detail of the element, eg: double, float.
 **/
class Partitioner_cscv {
    Partitioner_cscv();
    Partitioner_cscv(const Partitioner_cscv&);

    Img_param m_img_param;  // the CT img param

    int m_x_group_size, m_y_group_size, m_angle_group_size;  // group size
    int m_pxg_size = -1;  // for cscv only
    int m_img_x_block_count, m_img_y_block_count, m_block_angle;  // full block count on each dimension

    int m_angle_part, m_x_part, m_y_part;  // part count, notice that the full block count is their product
    std::vector<int> m_img_x_block_id_offsets, m_img_y_block_id_offsets, m_angle_block_id_offsets;  // offsets of parts

    void build_img_grouping();

public:
    Partitioner_cscv(Img_param img_param, int angle_group, int x_group, int y_group, int angle_part, int x_part, int y_part);

    template <class Element_type>
    Part<Element_type>* build_part(int part_id);
    template <class Element_type>
    void fill_part(Part<Element_type>* part);
    // template <class Element_type>
    // std::vector<Part<Element_type>*> build_parts();

    int get_logical_part_count() const { return m_angle_part * m_x_part * m_y_part; }

    std::string get_summary_string() const;

    const Img_param& get_img_param() const { return m_img_param; }
    int get_x_group_size() const { return m_x_group_size; }
    int get_y_group_size() const { return m_y_group_size; }
    int get_angle_group_size() const { return m_angle_group_size; }

    Partition_result_cscv get_partition_result();

    template <class Element_type>
    int get_min_stack_size_mb() {
        size_t ret = 0;

        for (int angle_part_id = 0; angle_part_id < m_angle_part; angle_part_id++) {
            int angle_count = (m_angle_block_id_offsets[angle_part_id + 1] - m_angle_block_id_offsets[angle_part_id]) * m_angle_group_size;
            size_t sz = sizeof(Element_type) * angle_count * m_img_param.m_num_bin;
            if (ret < sz)
                ret = sz;
        }

        return div_and_ceil(ret, 1024 * 1024);
    }

    // from highest dim to lowest dim
    enum class Block_order {
        ANGLE_X_Y,
        ANGLE_Y_X,
        X_ANGLE_Y,
        X_Y_ANGLE,
        Y_ANGLE_X,
        Y_X_ANGLE,
    };

    std::map<Block_order, std::string> m_order_label_map;

    void set_block_order(Block_order order);
    void set_pxg_size(int pxg_size);
    void set_computation_config(Computation_config comp_cfg) { m_comp_cfg = comp_cfg; }

private:
    Block_order m_block_order = (Block_order)(-1);
    Computation_config m_comp_cfg;
};

/**
 * Perform data generation and computation (including validation)
 **/
template <class Element_type>
class Data_holder {
    Img_param m_img_param;
    System_matrix_generator* m_mtx_generator = nullptr;
    System_matrix_converter_cscv<Element_type>* m_cscv_converter = nullptr;
    Partition_result_cscv m_partition_result;
    Partitioner_cscv* m_partitioner = nullptr;
    Result_table<Element_type> m_y_result_table, m_x_result_table;

    std::vector<Part<Element_type>*> m_parts;

    std::vector<Ring_fake_barrier*> m_y_reduction_barriers;

    int m_part_count;

    int m_num_threads = 1;

    COO_matrix_buffer<Element_type>* m_coo_full = nullptr;
    CSC_matrix<Element_type>* m_csc_full = nullptr;
    CSR_matrix<Element_type>* m_csr_full = nullptr;
    sparse_matrix_t* m_csc_mkl_full = nullptr;
    sparse_matrix_t* m_csr_mkl_full = nullptr;
    sparse_matrix_t* m_csc_trans_mkl_full = nullptr;
    sparse_matrix_t* m_csr_trans_mkl_full = nullptr;

    Image_CT<Element_type> *m_x_input = nullptr;
    Image_CT<Element_type> *m_x_tmp = nullptr;
    Dense_vector<Element_type> *m_y_tmp = nullptr;
    Dense_vector<Element_type> *m_y_std = nullptr;

    Result_table<Element_type> m_calc_y_timer_table, m_calc_x_timer_table, m_init_timer_table;

    Computation_config m_comp_cfg;

    // thread local timers

    // what may be important: max spin time, full spin ratio
    struct Thread_calculation_timer {
        double m_part_data_init_time = 0.0;
        double m_calculation_time = 0.0;
        double m_spin_time = 0.0;
        double m_part_data_reduction_time = 0.0;
    };

    std::vector<std::map<Result_type, Thread_calculation_timer> > m_thread_calculation_timers, m_thread_reverse_calculation_timers;  // size: nthreads

    void reduce_y_tmp_from_parts();

    // data generation status
    bool m_generated_full_coo = false, m_generated_full_csc = false, m_generated_full_csr = false;
    bool m_generated_part_coo = false, m_generated_part_csc = false, m_generated_part_csr = false;
    bool m_generated_block_coo = false, m_generated_block_csc = false, m_generated_block_csr = false, m_generated_block_cscvb = false;

    enum class Init_process_type {
        COO_PART,
        CSR_PART,
        CSC_PART,
        COO_FULL,
        CSR_FULL,
        CSC_FULL,
        COO_BLOCK,
        CSC_BLOCK,
        CSR_BLOCK,
        CSCVB_BLOCK,
    };

    // enum class Thread_timer_type {
    //     PART_DATA_INIT,
    //     COMPUTE_KERNEL,
    //     SPIN_AFTER_COMPUTE,
    //     PART_DATA_REDUCTION,
    // };

    std::map<Init_process_type, std::string> m_init_process_label_map;

    bool m_std_y_produced = false;

    OMP_thread_pool* m_thread_pool = nullptr;
    using Member_func = void(*)(Data_holder<Element_type>* param);

    template <uint8_t t_vec_angle, uint8_t t_px_group>
    void compute_blocks_cscvb_tea_y_ax_inner();
    template <uint8_t t_vec_angle>
    void compute_blocks_cscvb_tea_y_ax_px_group_expand();

    // tmp config for soa computing
    bool m_use_cat, m_use_mkl_csc;

    void thread_local_init();

    Result_type m_current_result_type;

    std::vector<Timers_instance_map> m_timers_maps_y_ax;

    void set_num_threads(int num_threads);
    void build_thread_local_parts();
    void build_global_arrs();

    // expense related
    uint64_t m_tea_iter_bytes_expense, m_cat_iter_bytes_expense;

    uint64_t m_original_nnz = 0;

public:
    Data_holder(const Img_param& img_param, const Partition_result_cscv& partition_result, Partitioner_cscv* partitioner,
                const Computation_config& m_comp_cfg, int nthreads);
    ~Data_holder();

    void one_loop_finished_y_ax();

    double init_full_coo();
    double init_full_csr();
    double init_full_csc();
    double init_parts_coo();
    double init_parts_csc();
    double init_parts_csr();
    double init_blocks_cscvb();
    double init_blocks_coo_debug_only();

    void free_parts_coo();
    void free_full_coo();

    double compute_full_coo_y_ax();
    double compute_full_csr_mkl_y_ax();
    double compute_full_csc_mkl_y_ax();
    double compute_parts_coo_y_ax();  // and get std result
    double compute_blocks_cscvb_tea_y_ax(bool use_cat);

    void estimate_cscvb_tea_cat_mem_expense();

    std::string get_workload_filename() const;
    void print_calculation_timer();
    void print_init_timer();
    void compare_y_and_print();
    void compare_x_and_print();

    void get_calculation_thread_timer_rough();

    std::string status_string() const;

    int get_max_cscv_block_bin_count() const;

    void estimate_block_arr_expense() const;

    friend class Partitioner_cscv;

    void free_system_matrix_gen();

    void validate_generation();
};


// #include "partition.tcc"
// #include "partition_init.tcc"
// #include "partition_y_ax.tcc"
// #include "partition_cscv.tcc"
// #include "part.tcc"
