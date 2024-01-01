#pragma once

#include "arch/naive_numa_util.hpp"
#include "base/logger.hpp"
#include <omp.h>

#include "cscv/partition.hpp"
#include "cscv/system_matrix_converter.hpp"

class Test_performer {
    int m_nthreads;
    int m_img_size, m_num_angle, m_num_bin;
    int m_global_img_size, m_global_img_x_start, m_global_img_y_start;
    double m_delta_angle;

    int m_img_x_part, m_img_y_part, m_angle_part;
    int m_img_x_group_size, m_img_y_group_size, m_angle_group_size;

    int m_pxg_size;

    bool m_skip_running;

    std::string m_logpath;
    std::string m_finish_signal_path;
    bool m_log_to_new_dir;

    Computation_config m_comp_cfg;

    enum class Float_precision {
        FLOAT,
        DOUBLE,
    };

    // coo_part will be run by default

    Partitioner_cscv::Block_order m_block_order;
    Float_precision m_float_precision;

    std::vector<std::pair<std::string, std::string> > get_args(int argc, char** argv, std::string help) const;

public:
    Test_performer(int argc, char** argv);

    void run();

    template <class Element_type>
    void run_inner();

private:
    void set_workload_id(int);
    void set_nthreads(int);

    std::string get_summary() const;
};
