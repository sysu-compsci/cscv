#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdarg>
#include <iostream>
#include <string>
#include <fstream>

#include "base/basic_definition.hpp"

class Logger {
    mutable std::string m_filepath;
    mutable std::ofstream* m_ofs = nullptr;

    static std::string get_dirname(const std::string& ori_dirname) {
        if (!check_if_exist(ori_dirname) || ori_dirname.size() == 0) {
            return ori_dirname;
        }

        for (int i = 0; i < 100; i++) {
            std::string final_dirname = strprintf("%s_%d", ori_dirname.c_str(), i);
            if (!check_if_exist(final_dirname)) {
                return final_dirname;
            }
        }
        return "";
    }

    ~Logger() {
        close_file();
    }

    static bool check_if_exist(std::string path) {
        struct stat info;

        if( stat( path.c_str(), &info ) != 0 )
            return false;
        else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows 
            return true;
        else
            return true;
    }

public:
    static std::string& dirname_ref() {
        static std::string dirname;
        return dirname;
    }

    // thread local
    static const Logger& get_instance() {
        static __thread Logger instance;
        return instance;
    }

    bool is_active() const {
        return m_ofs != nullptr;
    }

    template <class T>
    void write(const T& val, bool to_stdout = false) const {
        if (to_stdout)
            std::cout << val;

        if (m_ofs == nullptr)
            return;

        *m_ofs << val;
    }

    void set_filename(const std::string& filename) const {
        if (dirname_ref().size() != 0) {
            std::string filepath = dirname_ref() + "/" + filename;

            if (filepath == m_filepath) {
                return;
            }

            close_file();

            // m_ofs = new std::ofstream(filepath.c_str(), std::ios_base::app);
            m_ofs = new std::ofstream(filepath.c_str());
            m_filepath = filepath;
        }
    }

    void close_file() const {
        if (m_ofs != nullptr) {
            delete m_ofs;
            m_ofs = nullptr;
        }
    }

    static void set_global_dir(const std::string& dirname, bool create) {
        // called by main thread
        std::string final_dirname;
        if (create) {
            final_dirname = get_dirname(dirname);
            if (final_dirname.size() != 0) {
                int not_ok = mkdir(final_dirname.c_str(), 0755);
                if (!not_ok) {
                    dirname_ref() = final_dirname;
                }
            }
        } else {
            final_dirname = dirname;
            int not_ok = mkdir(final_dirname.c_str(), 0755);
            dirname_ref() = final_dirname;
        }

    }
};
