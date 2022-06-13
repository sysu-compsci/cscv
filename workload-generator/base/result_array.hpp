#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// TODO: for intergral types, can get inergral max / min / median, and can get floating point average

template <class Element_type>
class Result_array {
    std::vector<Element_type> m_arr;

    static constexpr Element_type INVALID_RESULT = -2333.3333;

public:
    Element_type& at(int off) { return m_arr.at(off); }
    const Element_type& at(int off) const { return m_arr.at(off); }
    Element_type& operator [](int off) { return at(off); }
    const Element_type& operator [](int off) const { return at(off); }
    // int& operator[](size_t offset) { return m_coords[offset]; }
    void resize(int sz) { m_arr.resize(sz); }
    Result_array();
    Result_array(const std::vector<Element_type>&);
    void append_result(const Element_type& result);
    void push_back(const Element_type& result) { append_result(result); }
    Result_array<Element_type> truncate_both_ends(Element_type ratio) const;  // bad naming
    Element_type get_median() const;
    Element_type get_average() const;
    Element_type get_min() const;
    Element_type get_max() const;
    std::string get_detail() const;
    int size() const { return m_arr.size(); }
    void clear() { m_arr.clear(); }
};

template <class Element_type>
Result_array<Element_type>::Result_array() {}

template <class Element_type>
Result_array<Element_type>::Result_array(const std::vector<Element_type>& results) {
    m_arr = results;
}

template <class Element_type>
void Result_array<Element_type>::append_result(const Element_type& result) {
    m_arr.emplace_back(result);
}

template <class Element_type>
Result_array<Element_type> Result_array<Element_type>::truncate_both_ends(Element_type ratio) const {
    if (ratio >= 0.5 || ratio <= 0)
        return *this;

    size_t off = ratio * m_arr.size();
    size_t l = off, r = m_arr.size() - off;

    std::vector<Element_type> sub_results = std::vector<Element_type>(m_arr.begin() + l, m_arr.begin() + r);
    return Result_array(move(sub_results));
}

template <class Element_type>
Element_type Result_array<Element_type>::get_median() const {
    if (m_arr.size() == 0)
        return INVALID_RESULT;

    auto results = m_arr;
    sort(results.begin(), results.end());

    // TODO: opt this, do not use branch here
    if (results.size() % 2 != 0)
        return results[results.size() / 2];
    else
        return (results[results.size() / 2] + results[-1 + results.size() / 2]) / 2.0;
}

template <class Element_type>
Element_type Result_array<Element_type>::get_average() const {
    if (m_arr.size() == 0)
        return INVALID_RESULT;

    Element_type ret = 0.0;

    for (Element_type result : m_arr)
        ret += result;

    return ret / m_arr.size();
}

template <class Element_type>
Element_type Result_array<Element_type>::get_min() const {
    if (m_arr.size() == 0)
        return INVALID_RESULT;

    Element_type const & (*min) (Element_type const &, Element_type const &) = std::min<Element_type>;
    return accumulate(m_arr.begin() + 1, m_arr.end(), m_arr.front(), min);
}


template <class Element_type>
Element_type Result_array<Element_type>::get_max() const {
    if (m_arr.size() == 0)
        return INVALID_RESULT;

    Element_type const & (*max) (Element_type const &, Element_type const &) = std::max<Element_type>;
    return accumulate(m_arr.begin() + 1, m_arr.end(), m_arr.front(), max);
}

template <class Element_type>
std::string Result_array<Element_type>::get_detail() const {
    std::stringstream ss;
    ss << "[";
    for (Element_type result : m_arr)
        ss << ", " << result;
    ss << "]";
    return ss.str();
}
