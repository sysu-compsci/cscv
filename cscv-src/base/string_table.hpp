#pragma once

#include <map>

#include "base/basic_definition.hpp"

/**
 * When printed, the mtx cols should be aligned.
 **/
class String_table {
private:
    typedef std::map<std::string, std::string*> String_map;

    // to order the row & col title manually
    std::vector<std::string> m_row_titles, m_col_titles;

    std::map<std::string, String_map> m_rows;
    std::map<std::string, String_map> m_cols;

    static std::string to_string_inner(std::vector<std::vector<std::string> >&);

    std::string m_table_name;

    template <class T>
    std::string value_to_string(const T& value) const;
    std::string value_to_string(const bool& value) const {
        if (value)
            return "true";
        else
            return "false";
    }

public:
    String_table(std::string table_name) : m_table_name(table_name) {}
    ~String_table();

    std::string to_string() const;
    // std::string to_string_transposed() const;

    template <class T>
    void set_value(const std::string& row, const std::string& col, const T& value);

    std::string& at(const std::string& row, const std::string& col);

    void append_row_title(const std::string& row_title);
    void append_col_title(const std::string& col_title);
    void append_row_titles(const std::vector<std::string>& row_titles);
    void append_col_titles(const std::vector<std::string>& col_titles);
};

template <class T>
std::string String_table::value_to_string(const T& value) const {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <class T>
void String_table::set_value(const std::string& row, const std::string& col, const T& value) {
    std::string* str;

    if (m_rows.count(row) == 0)
        m_row_titles.push_back(row);
    if (m_cols.count(col) == 0)
        m_col_titles.push_back(col);

    if (m_rows[row].count(col) == 0) {
        str = new std::string;
        m_rows[row][col] = str;
        m_cols[col][row] = str;
    } else {
        str = &at(row, col);
    }

    *str = value_to_string(value);
}

#define rt_set_row_value(table_ptr, col, value) (table_ptr)->set_value((#value), (col), (value))
