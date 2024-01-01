#include "cscv.hpp"

template <class Element_type>
CSCVB_matrix_block<Element_type>::~CSCVB_matrix_block() {
    if (m_yt_buffer)
        delete m_yt_buffer;
}

static inline std::string get_timer_map_str_for_particular_key(const Timers_instance_map& _timer_map, SPMV_direction direction, Result_type framework) {
    Timers_instance_map timer_map = _timer_map;

    String_table table(c_result_label_map.at((Result_type)framework) + (((int)direction == 0) ? " y = Ax" : " x = ATy"));

    for (auto& p_tid : timer_map) {
        for (auto& p_direction : p_tid.second.m_timers) {
            if (p_direction.first != (int)direction)
                continue;
            for (auto& p_framework : p_direction.second) {
                if (p_framework.first != (int)framework)
                    continue;
                uint64_t sum = 0;
                for (auto& p_process : p_framework.second) {
                    sum += p_process.second;
                }
                if (sum != 0)
                    p_framework.second[(int)Timer_type::SUM] = sum;
            }
        }
    }

    // collapse to double
    std::map<int, Three_layer_timer<double> > timer_map_double;

    for (auto& p_tid : timer_map) {
        for (auto& p_direction : p_tid.second.m_timers) {
            if (p_direction.first != (int)direction)
                continue;
            for (auto& p_framework : p_direction.second) {
                if (p_framework.first != (int)framework)
                    continue;
                for (auto& p_process : p_framework.second) {
                    timer_map_double[p_tid.first].m_timers[p_direction.first][p_framework.first][p_process.first] =
                                     p_process.second * 1.0 / p_framework.second[(int)Timer_type::SUM];
                }
            }
        }
    }

    bool empty = true;

    for (auto p_tid : timer_map_double) {
        for (auto p_direction : p_tid.second.m_timers) {
            for (auto p_framework : p_direction.second) {
                for (auto p_process : p_framework.second) {
                    int direction_key = p_direction.first;
                    ASSERT_AND_PRINTF(direction_key == (int)SPMV_direction::Y_AX, "");
                    std::string direction_str = (p_direction.first == 0) ? "y = Ax" : "x = ATy";
                    int framework_key = p_framework.first;
                    std::string framework_str = c_result_label_map.at((Result_type)framework_key);
                    std::string process_str = c_timer_str_map.at((Timer_type)p_process.first);

                    std::string thread_id_str = strprintf("%d", p_tid.first);
                    table.set_value(thread_id_str, process_str, p_process.second);

                    empty = false;
                }
            }
        }
    }

    std::stringstream ss;
    ss << table.to_string();

    return ss.str();
}

static inline std::string get_timer_map_str(const Timers_map& _timer_map) {
    Timers_map timer_map = _timer_map;

    std::map<int, String_table*> tables_maps[2];

    for (auto& p_tid : timer_map) {
        for (auto& p_direction : p_tid.second->m_timers) {
            for (auto& p_framework : p_direction.second) {
                uint64_t sum = 0;
                for (auto& p_process : p_framework.second) {
                    sum += p_process.second;
                }
                if (sum != 0)
                    p_framework.second[(int)Timer_type::SUM] = sum;
            }
        }
    }

    // collapse to double
    std::map<int, Three_layer_timer<double> > timer_map_double;

    for (auto& p_tid : timer_map) {
        for (auto& p_direction : p_tid.second->m_timers) {
            for (auto& p_framework : p_direction.second) {
                for (auto& p_process : p_framework.second) {
                    timer_map_double[p_tid.first].m_timers[p_direction.first][p_framework.first][p_process.first] =
                                     p_process.second * 1.0 / p_framework.second[(int)Timer_type::SUM];
                }
            }
        }
    }

    for (auto p_tid : timer_map_double) {
        for (auto p_direction : p_tid.second.m_timers) {
            for (auto p_framework : p_direction.second) {
                for (auto p_process : p_framework.second) {
                    int direction_key = p_direction.first;
                    ASSERT_AND_PRINTF(direction_key == (int)SPMV_direction::Y_AX, "");
                    std::string direction_str = (p_direction.first == 0) ? "y = Ax" : "x = ATy";
                    int framework_key = p_framework.first;
                    std::string framework_str = c_result_label_map.at((Result_type)framework_key);
                    std::string process_str = c_timer_str_map.at((Timer_type)p_process.first);

                    if (tables_maps[direction_key].count(framework_key) == 0) {
                        tables_maps[direction_key][framework_key] = new String_table(framework_str + " " + direction_str);
                    }

                    std::string thread_id_str = strprintf("%d", p_tid.first);
                    tables_maps[direction_key][framework_key]->set_value(thread_id_str, process_str, p_process.second);
                }
            }
        }
    }

    std::stringstream ss;

    for (int i = 0; i < 2; i++) {
        for (auto p : tables_maps[i]) {
            ss << p.second->to_string();

            delete p.second;
        }
    }

    return ss.str();
}
