#pragma once

class Test_performer;

class Flag {
    bool m_dump_figure = false;

    Flag(const Flag&);
    Flag() {}

public:
    bool dump_figure() const { return m_dump_figure; }

    static Flag& get_instance() {
        static Flag instance;
        return instance;
    }

    friend Test_performer;
};

constexpr uint8_t c_pxg_max_bin = 14;
constexpr uint16_t c_block_max_bin = 48;
