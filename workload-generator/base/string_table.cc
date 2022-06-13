#include "string_table.hpp"

using namespace std;

String_table::~String_table() {
    for (auto row : m_rows) {
        for (auto p : row.second) {
            delete p.second;
        }
    }
}

string String_table::to_string_inner(vector<vector<string> >& cols) {
    for (auto& col : cols) {
        align_strings_by_char(&col.at(0), col.size(), ' ');
    }

    stringstream ss;

    string row_spliter, row_spliter_edge, first_row;

    for (int r = 0; r < cols[0].size(); r++) {
        for (int c = 0; c < cols.size(); c++) {
            ss << cols[c][r];
            if (c == 0 || c == cols.size() - 2) {
                ss << " || ";
            } else {
                ss << " | ";
            }
        }
        if (r == 0) {
            first_row = ss.str();
            int first_row_sz = first_row.size();
            row_spliter = extended_string_by_char("", first_row_sz, '-');
            row_spliter[row_spliter.size() - 1] = ' ';
            row_spliter_edge = row_spliter;
            row_spliter[row_spliter.size() - 2] = '|';

            ss = stringstream();
            ss << row_spliter_edge << endl << first_row << endl << row_spliter;
        } else if (r == cols[0].size() - 1) {
            ss << endl << row_spliter;
        }

        ss << endl;
    }
    ss << first_row << endl << row_spliter_edge << endl;

    return ss.str();
}

string String_table::to_string() const {
    vector<vector<string> > cols;

    cols.resize(m_col_titles.size() + 2);

    cols[0].push_back(m_table_name);
    cols[0].insert(cols[0].end(), m_row_titles.begin(), m_row_titles.end());
    cols[m_col_titles.size() + 1].push_back(m_table_name);
    cols[m_col_titles.size() + 1].insert(cols[m_col_titles.size() + 1].end(), m_row_titles.begin(), m_row_titles.end());

    for (int c = 0; c < m_col_titles.size(); c++) {
        const string& col_title = m_col_titles[c];
        const String_map& col_map = m_cols.at(col_title);

        cols[c + 1].push_back(col_title);
        for (auto row_title : m_row_titles) {
            if (col_map.count(row_title) == 0) {
                cols[c + 1].push_back("");
            } else {
                cols[c + 1].push_back(*col_map.at(row_title));
            }
        }
    }

    return to_string_inner(cols);
}

string& String_table::at(const std::string& row, const std::string& col) {
    ASSERT_AND_PRINTF(m_rows.count(row) != 0, "row %s does not exist!\n", row.c_str());
    ASSERT_AND_PRINTF(m_rows.at(row).count(col) != 0, "in row %s, col %s does not exist!\n", row.c_str(), col.c_str());

    return *m_rows.at(row).at(col);
}

void String_table::append_col_title(const string& col_title) {
    if (m_cols.count(col_title) == 0) {
        m_col_titles.push_back(col_title);
        m_cols[col_title];
    }
}

void String_table::append_row_title(const string& row_title) {
    if (m_rows.count(row_title) == 0) {
        m_row_titles.push_back(row_title);
        m_rows[row_title];
    }
}

void String_table::append_col_titles(const vector<string>& col_titles) {
    for (auto col_title : col_titles)
        append_col_title(col_title);
}

void String_table::append_row_titles(const vector<string>& row_titles) {
    for (auto row_title : row_titles)
        append_row_title(row_title);
}
