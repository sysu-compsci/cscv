#include "test_performer.hpp"

#include "base/partition_kernel.hpp"
#include "cscv/flags.hpp"

using namespace std;

vector<pair<string, string> > Test_performer::get_args(int argc, char** argv, string help) const {
    vector<pair<string, string> > ret;

    ASSERT_AND_PRINTF((argc - 1) % 2 == 0, "wrong argc %d, should be odd\n%s\n", argc, help.c_str());
    argc -= 1;

    for (int i = 1; i < argc; i+= 2) {
        string option = argv[i];
        string value = argv[i + 1];
        ASSERT_AND_PRINTF(option.size() > 1 && option.at(0) == '-', "invalid option specifier %s\nusage:\n%s\n", option.c_str(), help.c_str());
        option.erase(0, 1);
        ret.emplace_back(option, value);
    }

    return ret;
}

Test_performer::Test_performer(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    string help_message = "	-t nthreads <int> (default: mkl_max_thread)\n\
	-w: workload-id <1/2/3/13/14> (default: 1)\n\
    -img_size: <int> (default: 128)\n\
    -num_angle: <int> (default: 60)\n\
    -num_bin: <int> (default: 192)\n\
    -delta_angle: <float> (default: 0.375)\n\
	-xp/yp/ap: partition count <int> (currently invalid, with default value in each nthreads)\n\
	-ba/bx/by: block granurality <int> (default: 16 16 16)\n\
	-l: loop <int> (default: 1000)\n\
	-bo: block order <axy, …> (default: axy)\n\
	-p: precision <f/d> (default: f)\n\
	-r: test to run. accumulative <cscvb_cat, cscvb_tea, mkl_csr_full, mkl_csc_full, coo_full, all, all_all>\n\
    -reverse: run reverse computation or not <0/1>\n\
    -reverse-only: run reverse computation or not <0/1>\n\
    -pxg: pxg size for tea <int> (default: 2)\n\
    -logpath: the directory of the log files <string>\n\
    -log_to_new_dir: create a new directory for logging <0/1>\n\
    -dump_figure: dump log for figuring <0/1>\n\
    -global_img_size: <int> (default: img_size)\n\
    -global_img_x_start: <int> (default: 0)\n\
    -global_img_y_start: <int> (default: 0)\n\
    -gen_constant: <0/1> (default: 0)\n\
    -print_imba: <0/1> (default: 0)\n\
    -disable_validation: <0/1> (default: 0)\n\
    -skip_running: <0/1> (default: 0)\n\
    -stack_size: pthread stack size in MB <int> (default: -1 (unset))\n\
    -numa_mempool_mb: size of memory pool in MB (on each NUMA node) <int> (default: -1 (disabled)) (DO NOT USE THIS!)\n\
    -f: the path of finishing signal file <string>\n\
    -block_arr_size: print the total size of block arr <0/1>\n\
    -validate_generation_only: only validate the coherence of full, part and block coo <0(default)/1>\n\
    ";

    vector<pair<string, string> > args = get_args(argc, argv, help_message);

    // setting default value

    int nthreads = mkl_get_max_threads();
    int x_part = -1, y_part = -1, angle_part = 1;
    int workload_id = 1;
    int block_x = 16, block_y = 16, block_angle = 16;
    m_comp_cfg.m_loop_count = 1000;
    int img_size = -1, num_angle = -1, num_bin = -1;
    double delta_angle = 0;
    Float_precision float_precision = Float_precision::FLOAT;
    Partitioner_cscv::Block_order block_order = Partitioner_cscv::Block_order::ANGLE_X_Y;
    m_logpath = "";
    m_log_to_new_dir = false;
    m_skip_running = false;

    m_comp_cfg.m_run_mkl_csc_full = false, m_comp_cfg.m_run_mkl_csr_full = false, m_comp_cfg.m_run_coo_full = false;
    m_comp_cfg.m_run_cscvb_tea = false, m_comp_cfg.m_run_cscvb_cat = false;
    m_comp_cfg.m_run_normal = true;
    m_comp_cfg.m_print_thread_imba = false;

    m_comp_cfg.m_gen_constant = false;
    m_comp_cfg.m_disable_validation = false;

    m_comp_cfg.m_stack_size_mb = -1;

    m_comp_cfg.m_print_cscv_expansion_only = false;
    m_comp_cfg.m_ref_px_x_offset = -1;
    m_comp_cfg.m_ref_px_y_offset = -1;

    m_pxg_size = 2;

    int global_img_size = -1;
    int global_img_x_start = -1;
    int global_img_y_start = -1;

    m_global_img_x_start = m_global_img_y_start = 0;
    m_global_img_size = -1;

    m_comp_cfg.m_mempool_mb = -1;

    m_comp_cfg.m_print_block_arr_size = false;

    m_comp_cfg.m_validate_generation_only = false;
    m_comp_cfg.m_snapshot_for_full = false;

    for (auto p : args) {
        if (p.first == "t") {
            nthreads = atoi(p.second.c_str());
        } else if (p.first == "w") {
            workload_id = atoi(p.second.c_str());
        } else if (p.first == "img_size") {
            img_size = atoi(p.second.c_str());
        } else if (p.first == "num_angle") {
            num_angle = atoi(p.second.c_str());
        } else if (p.first == "num_bin") {
            num_bin = atoi(p.second.c_str());
        } else if (p.first == "delta_angle") {
            delta_angle = atof(p.second.c_str());
        } else if (p.first == "xp") {
            x_part = atoi(p.second.c_str());
        } else if (p.first == "yp") {
            y_part = atoi(p.second.c_str());
        } else if (p.first == "ap") {
            angle_part = atoi(p.second.c_str());
        } else if (p.first == "bx") {
            block_x = atoi(p.second.c_str());
        } else if (p.first == "by") {
            block_y = atoi(p.second.c_str());
        } else if (p.first == "ba") {
            block_angle = atoi(p.second.c_str());
        } else if (p.first == "l") {
            m_comp_cfg.m_loop_count = atoi(p.second.c_str());
        } else if (p.first == "bo") {
            if (p.second == "axy") {
                block_order = Partitioner_cscv::Block_order::ANGLE_X_Y;
            } else if (p.second == "ayx") {
                block_order = Partitioner_cscv::Block_order::ANGLE_Y_X;
            } else if (p.second == "yax") {
                block_order = Partitioner_cscv::Block_order::Y_ANGLE_X;
            } else if (p.second == "yxa") {
                block_order = Partitioner_cscv::Block_order::Y_X_ANGLE;
            } else if (p.second == "xay") {
                block_order = Partitioner_cscv::Block_order::X_ANGLE_Y;
            } else if (p.second == "xya") {
                block_order = Partitioner_cscv::Block_order::X_Y_ANGLE;
            } else {
                ASSERT_AND_PRINTF(false, "invalid block order %s\n%s\n", p.second.c_str(), help_message.c_str());
            }
        } else if (p.first == "p") {
            if (p.second == "f") {
                float_precision = Float_precision::FLOAT;
            } else if (p.second == "d") {
                float_precision = Float_precision::DOUBLE;
            } else {
                ASSERT_AND_PRINTF(false, "invalid float precision %s\n%s\n", p.second.c_str(), help_message.c_str());
            }
        } else if (p.first == "r") {
            if (p.second == "mkl_csr_full" || p.second == "mkl_csr") {
                m_comp_cfg.m_run_mkl_csr_full = true;
            } else if (p.second == "mkl_csc_full" || p.second == "mkl_csc") {
                m_comp_cfg.m_run_mkl_csc_full = true;
            } else if (p.second == "coo_full") {
                m_comp_cfg.m_run_coo_full = true;
            } else if (p.second == "cscvb_tea" || p.second == "tea") {
                m_comp_cfg.m_run_cscvb_tea = true;
            } else if (p.second == "cscvb_cat" || p.second == "cat") {
                m_comp_cfg.m_run_cscvb_cat = true;
            } else if (p.second == "all") {
                m_comp_cfg.m_run_cscvb_tea = true;
                m_comp_cfg.m_run_cscvb_cat = true;
            } else if (p.second == "all_all") {
                m_comp_cfg.m_run_cscvb_tea = true;
                m_comp_cfg.m_run_cscvb_cat = true;
                m_comp_cfg.m_run_mkl_csc_full = true;
                m_comp_cfg.m_run_mkl_csr_full = true;
            } else {
                ASSERT_AND_PRINTF(false, "invalid test name %s\n%s\n", p.second.c_str(), help_message.c_str());
            }
        } else if (p.first == "pxg" || p.first == "vxg") {
            m_pxg_size = atoi(p.second.c_str());
        } else if (p.first == "logpath") {
            m_logpath = p.second;
        } else if (p.first == "log_to_new_dir") {
            int identifier = atoi(p.second.c_str());
            m_log_to_new_dir = (identifier != 0);
        } else if (p.first == "dump_figure") {
            int identifier = atoi(p.second.c_str());
            Flag::get_instance().m_dump_figure = (identifier != 0);
        } else if (p.first == "global_img_size") {
            global_img_size = atoi(p.second.c_str());
        } else if (p.first == "global_img_x_start") {
            global_img_x_start = atoi(p.second.c_str());
        } else if (p.first == "global_img_y_start") {
            global_img_y_start = atoi(p.second.c_str());
        } else if (p.first == "gen_constant") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_gen_constant = (identifier != 0);
        } else if (p.first == "print_imba") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_print_thread_imba = (identifier != 0);
        } else if (p.first == "disable_validation") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_disable_validation = (identifier != 0);
        } else if (p.first == "skip_running") {
            int identifier = atoi(p.second.c_str());
            m_skip_running = (identifier != 0);
        } else if (p.first == "stack_size") {
            m_comp_cfg.m_stack_size_mb = atoi(p.second.c_str());
        } else if (p.first == "print_cscv_expansion_only") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_print_cscv_expansion_only = (identifier != 0);
        } else if (p.first == "ref_px_x_offset") {
            m_comp_cfg.m_ref_px_x_offset = atoi(p.second.c_str());
        } else if (p.first == "ref_px_y_offset") {
            m_comp_cfg.m_ref_px_y_offset = atoi(p.second.c_str());
        } else if (p.first == "numa_mempool_mb") {
            m_comp_cfg.m_mempool_mb = atoi(p.second.c_str());
        } else if (p.first == "f") {
            m_finish_signal_path = p.second;
        } else if (p.first == "print_block_arr_size") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_print_block_arr_size = (identifier != 0);
        } else if (p.first == "validate_generation_only") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_validate_generation_only = (identifier != 0);
        } else if (p.first == "snapshot_for_full") {
            int identifier = atoi(p.second.c_str());
            m_comp_cfg.m_snapshot_for_full = (identifier != 0);
        } else {
            ASSERT_AND_PRINTF(false, "unknown keyword %s !!\n%s\n", p.first.c_str(), help_message.c_str());
        }
    }

    Logger::set_global_dir(m_logpath, m_log_to_new_dir);

    // set workload id
    set_workload_id(workload_id);

    if (img_size != -1)
        m_img_size = img_size;
    if (num_angle != -1)
        m_num_angle = num_angle;
    if (num_bin != -1)
        m_num_bin = num_bin;
    if (delta_angle != 0)
        m_delta_angle = delta_angle;

    PRINTF("m global img size = %d\n", m_global_img_size);

    if (global_img_size != -1) {
        m_global_img_size = global_img_size;
    } else {
        if (m_global_img_size == -1) {
            m_global_img_size = m_img_size;
        }
    }
    PRINTF("m global img size = %d\n", m_global_img_size);

    if (global_img_x_start != -1)
        m_global_img_x_start = global_img_x_start;

    if (global_img_y_start != -1)
        m_global_img_y_start = global_img_y_start;

    // set nthreads
    set_nthreads(nthreads);

    // set partition count
    if (x_part != -1 && y_part != -1 && angle_part != -1) {
        m_img_x_part = x_part;
        m_img_y_part = y_part;
        m_angle_part = angle_part;
    }

    // set block size
    m_img_x_group_size = block_x;
    m_img_y_group_size = block_y;
    m_angle_group_size = block_angle;

    // set other params
    m_float_precision = float_precision;
    m_block_order = block_order;

    ASSERT_AND_PRINTF(m_global_img_size >= m_img_size + m_global_img_x_start, "%d < %d + %d\n", m_global_img_size, m_img_size, m_global_img_x_start);
    ASSERT_AND_PRINTF(m_global_img_size >= m_img_size + m_global_img_y_start, "%d < %d + %d\n", m_global_img_size, m_img_size, m_global_img_y_start);
}

void Test_performer::set_workload_id(int workload_id) {
    if (workload_id == 1) {
        m_img_size = 128;
        m_num_angle = 60;
        m_num_bin = 192;
        m_delta_angle = 0.375;
    } else if (workload_id == 2) {
        m_img_size = 1024;
        m_num_angle = 64;
        m_num_bin = 1460;
        m_delta_angle = 0.375;
    } else if (workload_id == 3) {
        m_img_size = 1024;
        m_num_angle = 480;
        m_num_bin = 1460;
        m_delta_angle = 0.375;
    } else if (workload_id == 13) {
        m_img_size = 1024;
        m_num_angle = 64;
        m_num_bin = 5840;
        m_delta_angle = 0.1875;
        m_global_img_x_start = 1024;
        m_global_img_y_start = 1024;
        m_global_img_size = 4096;
    } else if (workload_id == 14) {
        m_img_size = 1024;
        m_num_angle = 480;
        m_num_bin = 5840;
        m_delta_angle = 0.1875;
        m_global_img_x_start = 1024;
        m_global_img_y_start = 1024;
        m_global_img_size = 4096;
    } else {
        PRINTF("invalid workload id %d\n", workload_id);
        set_workload_id(1);
    }
}

void Test_performer::set_nthreads(int nthreads) {
    m_angle_part = 1;
    pair<int, int> img_partition = checker_2d_dim_create(nthreads, 1.0);
    m_img_x_part = img_partition.first;
    m_img_y_part = img_partition.second;

    m_nthreads = nthreads;
}

void Test_performer::run() {
    if (m_float_precision == Float_precision::FLOAT) {
        run_inner<float>();
    } else if (m_float_precision == Float_precision::DOUBLE) {
        run_inner<double>();
    } else {
        ASSERT_AND_PRINTF(false, "???\n");
    }
}

string Test_performer::get_summary() const {
    String_table table("test performer summary");

    const map<Partitioner_cscv::Block_order, string> order_label_map {
        {Partitioner_cscv::Block_order::ANGLE_X_Y, "ANGLE_X_Y"},
        {Partitioner_cscv::Block_order::ANGLE_Y_X, "ANGLE_Y_X"},
        {Partitioner_cscv::Block_order::X_ANGLE_Y, "X_ANGLE_Y"},
        {Partitioner_cscv::Block_order::X_Y_ANGLE, "X_Y_ANGLE"},
        {Partitioner_cscv::Block_order::Y_ANGLE_X, "Y_ANGLE_X"},
        {Partitioner_cscv::Block_order::Y_X_ANGLE, "Y_X_ANGLE"},
    };

    string block_order = order_label_map.at(m_block_order);

    const map<Float_precision, string> fp_map {
        {Float_precision::DOUBLE, "DOUBLE"},
        {Float_precision::FLOAT, "FLOAT"},
    };

    string float_precision = fp_map.at(m_float_precision);

    rt_set_row_value(&table, "", m_nthreads);
    rt_set_row_value(&table, "", m_img_size);
    rt_set_row_value(&table, "", m_global_img_size);
    rt_set_row_value(&table, "", m_global_img_x_start);
    rt_set_row_value(&table, "", m_global_img_y_start);
    rt_set_row_value(&table, "", m_num_angle);
    rt_set_row_value(&table, "", m_num_bin);
    rt_set_row_value(&table, "", m_delta_angle);
    rt_set_row_value(&table, "", m_img_x_part);
    rt_set_row_value(&table, "", m_img_y_part);
    rt_set_row_value(&table, "", m_angle_part);
    rt_set_row_value(&table, "", m_img_x_group_size);
    rt_set_row_value(&table, "", m_img_y_group_size);
    rt_set_row_value(&table, "", m_angle_group_size);

    rt_set_row_value(&table, "", m_comp_cfg.m_run_mkl_csr_full);
    rt_set_row_value(&table, "", m_comp_cfg.m_run_mkl_csc_full);

    rt_set_row_value(&table, "", m_comp_cfg.m_run_cscvb_tea);
    rt_set_row_value(&table, "", m_comp_cfg.m_run_cscvb_cat);

    rt_set_row_value(&table, "", m_comp_cfg.m_run_normal);

    rt_set_row_value(&table, "", m_pxg_size);

    rt_set_row_value(&table, "", Flag::get_instance().m_dump_figure);

    rt_set_row_value(&table, "", m_logpath);
    rt_set_row_value(&table, "", m_log_to_new_dir);

    rt_set_row_value(&table, "", block_order);
    rt_set_row_value(&table, "", float_precision);

    rt_set_row_value(&table, "", m_comp_cfg.m_gen_constant);
    rt_set_row_value(&table, "", m_comp_cfg.m_print_thread_imba);

    rt_set_row_value(&table, "", m_comp_cfg.m_disable_validation);

    rt_set_row_value(&table, "", m_skip_running);

    rt_set_row_value(&table, "", m_comp_cfg.m_stack_size_mb);

    rt_set_row_value(&table, "", m_comp_cfg.m_print_cscv_expansion_only);

    rt_set_row_value(&table, "", m_comp_cfg.m_ref_px_x_offset);
    rt_set_row_value(&table, "", m_comp_cfg.m_ref_px_y_offset);

    rt_set_row_value(&table, "", m_comp_cfg.m_mempool_mb);

    return table.to_string();
}
