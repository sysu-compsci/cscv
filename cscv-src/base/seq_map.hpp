#pragma once

#include "assert.h"

#include <vector>

template <class Key_type, class Value_type>
class Seq_map{
private:
    std::vector<std::pair<Key_type, Value_type> > m_list;
public:
    int count(const Key_type& key) const {
        for (auto p : m_list)
            if (p.first == key)
                return 1;
        return 0;
    }

    Value_type& at(const Key_type& key) {
        for (auto& p : m_list) {
            if (p.first == key) {
                return p.second;
            }
        }
        assert(false);
    }

    void insert(const Key_type& key, const Value_type& value) {
        if (count(key) == 0) {
            m_list.push_back(std::make_pair(key, value));
        } else {
            at(key) = value;
        }
    }

    const std::vector<std::pair<Key_type, Value_type> > get_list() const {
        return m_list;
    }
};
