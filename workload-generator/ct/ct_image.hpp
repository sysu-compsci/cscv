#pragma once

#include "base/basic_definition.hpp"
#include "data/data_container.hpp"

/**
 * This class is implemented to store a image or subimage for CT.
 **/
template <class Element_type>
struct Image_CT {
    int m_x, m_y;
    // Element_type* m_data;
    Dense_vector<Element_type>* m_vec;

    Image_CT(int x, int y) : m_x(x), m_y(y) {
        m_vec = new Dense_vector<Element_type>(x * y);
    }

    ~Image_CT() {
        delete m_vec;
    }

    Element_type& at(int x, int y) {
        ASSERT_AND_PRINTF(x >= 0 && x < m_x && y >= 0 && y < m_y, "invalid <%d, %d> in <%d %d>\n", x, y, m_x, m_y);
        return m_vec->at(x + y * m_x);
    }

    const Element_type& at(int x, int y) const {
        ASSERT_AND_PRINTF(x >= 0 && x < m_x && y >= 0 && y < m_y, "invalid <%d, %d> in <%d %d>\n", x, y, m_x, m_y);
        return m_vec->at(x + y * m_x);
    }
};
