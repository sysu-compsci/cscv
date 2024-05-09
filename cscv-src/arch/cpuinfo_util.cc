#include "cpuinfo_util.hpp"
#include <fstream>
#include <set>


using namespace std;

struct HtInfoTuple {
    int m_core_id, m_level, m_ht_id;

    HtInfoTuple() = default;
    HtInfoTuple(int core_id, int level, int ht_id) : m_core_id(core_id), m_level(level), m_ht_id(ht_id) {}
    bool operator<(const HtInfoTuple& other) const {
        if (m_level < other.m_level) {
            return true;
        } else if (m_level < other.m_level) {
            return false;
        } else if (m_core_id > other.m_core_id) {
            return true;
        } else if (m_core_id < other.m_core_id) {
            return false;
        } else if (m_ht_id > other.m_ht_id) {
            return true;
        } else {
            return false;
        }
    }
};

CPUInfoUtil::CPUInfoUtil() {
    FILE* fp;
    fp = fopen("/proc/cpuinfo", "r");

    char *buffer = nullptr;
    size_t len = 0;
    int nread;
    std::vector<int> raw_processor;
    while(nread = getline(&buffer,&len,fp) != -1) {
        int val = 0;

        char* cptr = strstr(buffer, ":");

        if (cptr != nullptr)
        while(*cptr != '\n' && *cptr != 0) {
            if (*cptr >= '0' && *cptr <= '9') {
                val *= 10;
                val += *cptr - '0';
            }
            cptr++;
        }
        if (strstr(buffer,"processor") != nullptr) {
            raw_processor.push_back(val);
        }
    }
    fclose(fp);

    m_global_ht_count = raw_processor.size();

    m_global_core_id_by_global_ht_id.resize(m_global_ht_count);

    // read ~/.dfchtinfo. with multiple lines, each with "HARDWARE_THREAD_ID HARDWARE_CORE_ID CORE_LEVEL"
    std::string home = getenv("HOME");
    ifstream ifs(home + "/.dfchtinfo");
    if (!ifs.is_open()) {
        cerr << "Failed to open ~/.dfchtinfo" << endl;
        exit(1);
    }
    std::vector<int> ht_ids, core_ids, core_levels;
    std::set<HtInfoTuple> core_id_with_level_set;
    std::map<int, int> core_level_map;
    int ht_id, core_id, core_level;
    while (ifs >> ht_id >> core_id >> core_level) {
        ht_ids.push_back(ht_id);
        core_ids.push_back(core_id);
        core_levels.push_back(core_level);

        if (core_level_map.count(core_id) == 0) {
            core_level_map[core_id] = core_level;
        } else {
            if (core_level_map[core_id] != core_level) {
                cerr << "Core level mismatch for core id " << core_id << endl;
            }
        }

        core_id_with_level_set.insert(HtInfoTuple(core_id, core_level, ht_id));
    }

    m_global_core_id_by_global_ht_id.resize(m_global_ht_count);

    // map to real core id
    int local_core_id = -1;
    int last_global_core_id = -1;
    int local_ht_id = -1;
    for (auto it : core_id_with_level_set) {
        if (it.m_core_id != last_global_core_id) {
            local_core_id++;
            last_global_core_id = it.m_core_id;

            m_local_core_id_by_global_core_id[it.m_core_id] = local_core_id;
            m_global_core_id_by_local_core_id[local_core_id] = it.m_core_id;
        }

        local_ht_id++;

        m_local_ht_id_by_global_ht_id[it.m_ht_id] = local_ht_id;
        m_global_ht_id_by_local_ht_id[local_ht_id] = it.m_ht_id;
        m_global_core_id_by_global_ht_id[it.m_ht_id] = it.m_core_id;
        m_global_core_level_by_global_core_id[it.m_core_id] = it.m_level;
    }

    // setup m_local_core_id_by_local_ht_id
    m_local_core_id_by_local_ht_id.resize(m_global_ht_count);
    for (int i = 0; i < m_global_ht_count; i++) {
        m_local_core_id_by_local_ht_id[i] = m_local_core_id_by_global_core_id[m_global_core_id_by_global_ht_id[i]];
    }
}
