#pragma once

#include <memory.h>

#include <map>

#include "base/basic_definition.hpp"
#include "base/result_array.hpp"
#include "base/string_table.hpp"

/**
 * In the result table, column 0 is the std result.
 **/
template <class Element_type>
class Result_table {
    struct Result_column {
        std::string m_label;
        // std::vector<Element_type> m_results;
        Result_array<Element_type> m_results;
    };

    std::map<int, Result_column> m_cols;

public:
    void set_label(int key, const std::string& label);  // col 0 is the std col
    void collect_data(int key, Element_type data);
    void collect_data_arr(int key, int count, Element_type* data);
    void clear_collected_data();  // to not clear the label

    std::string get_rela_to_std_string_for_col(int col_id, String_table& str_table) const;
    std::string get_summary_diff_string(const std::string& title) const;
    std::string get_summary_statistics_string(const std::string& title) const;
    int get_result_size(int key) const { return m_cols.at(key).m_results.size(); };

    const Result_column& get_column(int key) const { return m_cols.at(key); }
};

template <class Element_type>
void Result_table<Element_type>::set_label(int key, const std::string& label) {
    m_cols[key].m_label = label;
}

template <class Element_type>
void Result_table<Element_type>::collect_data(int key, Element_type data) {
    m_cols[key].m_results.push_back(data);
}

template <class Element_type>
void Result_table<Element_type>::collect_data_arr(int key, int count, Element_type* data) {
    int ori_size = m_cols[key].m_results.size();
    m_cols[key].m_results.resize(ori_size + count);
    memcpy(&m_cols[key].m_results.at(ori_size), data, count * sizeof(Element_type));
}

template <class Element_type>
void Result_table<Element_type>::clear_collected_data() {
    for (auto& p : m_cols) {
        p.second.m_results.clear();
        assert(p.second.m_results.size() == 0);
    }
}

template <class Element_type>
std::string Result_table<Element_type>::get_rela_to_std_string_for_col(int col_id, String_table& str_table) const {
    const Result_column& std_col = m_cols.at(0);
    const Result_column& current_col = m_cols.at(col_id);

    std::stringstream ss;

    if (std_col.m_results.size() != current_col.m_results.size()) {
        return "";
    }

    Element_type eps;

    if (std::is_integral<Element_type>())
        eps = 1;
    else
        eps = 0.000000001;

    Element_type diff_sum = 0;
    Element_type diff_max = 0;
    Element_type rela_diff_max = 0;
    Element_type max, min;
    Element_type sum = 0;

    for (int i = 0; i < std_col.m_results.size(); i++) {
        Element_type diff = std::abs(std_col.m_results[i] - current_col.m_results[i]);
        Element_type rela_diff = std::abs(std_col.m_results[i] - current_col.m_results[i]) / (std::abs(std_col.m_results[i] + current_col.m_results[i]) + eps);

        sum += current_col.m_results[i];
        diff_sum += diff;
        diff_max = std::max(diff_max, diff);
        rela_diff_max = std::max(rela_diff_max, rela_diff);
        if (i == 0) {
            max = min = current_col.m_results[i];
        } else {
            max = std::max(max, current_col.m_results[i]);
            min = std::min(min, current_col.m_results[i]);
        }
    }

    str_table.set_value(current_col.m_label, "sum", sum);
    str_table.set_value(current_col.m_label, "min", min);
    str_table.set_value(current_col.m_label, "max", max);
    str_table.set_value(current_col.m_label, "diff_sum", diff_sum);
    str_table.set_value(current_col.m_label, "diff_max", diff_max);
    str_table.set_value(current_col.m_label, "rela_diff_max", rela_diff_max);

    return ss.str();
}

template <class Element_type>
std::string Result_table<Element_type>::get_summary_diff_string(const std::string& title) const {
    ASSERT_AND_PRINTF(m_cols.count(0) != 0, "std col not found!\n");

    String_table str_table(title);

    for (auto p : m_cols) {
        get_rela_to_std_string_for_col(p.first, str_table);
    }

    return str_table.to_string();
}

template <class Element_type>
std::string Result_table<Element_type>::get_summary_statistics_string(const std::string& title) const {
    ASSERT_AND_PRINTF(m_cols.count(0) != 0, "std col not found!\n");

    String_table str_table(title);
    str_table.append_col_titles({"avg", "max", "median", "min", "acc"});

    const Result_column& std_col = m_cols.at(0);
    if (std_col.m_results.size() == 0)
        return "";

    for (auto p : m_cols) {
        const Result_column& current_col = p.second;
        if (current_col.m_results.size() == 0)
            continue;

        str_table.set_value(current_col.m_label, "avg", current_col.m_results.get_average());

        str_table.set_value(current_col.m_label, "median", current_col.m_results.get_median());
        str_table.set_value(current_col.m_label, "min", current_col.m_results.get_min());
        str_table.set_value(current_col.m_label, "max", current_col.m_results.get_max());
        str_table.set_value(current_col.m_label, "acc", std_col.m_results.get_min() * 1.0 / current_col.m_results.get_min());
    }

    return str_table.to_string();
}
