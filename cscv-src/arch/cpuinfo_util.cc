#include "cpuinfo_util.hpp"

using namespace std;

CPUInfoUtil::CPUInfoUtil() {
    FILE* fp;
    fp = fopen("/proc/cpuinfo", "r");

    char *buffer = nullptr;
    size_t len = 0;
    int nread;

    int siblings_val;
    int cpu_cores_val;

    map<int, map<int, int>> core_occu;  // core_occu[socket][core] = occupancy_count (starts from 0)

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
            raw_processor_.push_back(val);
        }

        if (strstr(buffer,"physical id") != nullptr) {
            raw_physical_id_.push_back(val);

            if (socket_proc_count_map_.count(val) == 0) {
                socket_proc_count_map_.insert(make_pair(val, 1));
                threads_by_socket_[val] = vector<int>();
                threads_by_socket_ht_[val] = vector<vector<int>>();
                core_occu[val] = map<int, int>();
            } else {
                socket_proc_count_map_[val]++;
            }
            threads_by_socket_[val].push_back(raw_processor_.back());
        }

        if (strstr(buffer,"siblings") != nullptr) {
            raw_siblings_.push_back(val); siblings_val = val;
        }

        if (strstr(buffer,"core id") != nullptr) {
            raw_core_id_.push_back(val);

            //notice that core id occurs after physical_id
            if (core_occu[raw_physical_id_.back()].count(val) == 0) {
                core_occu[raw_physical_id_.back()][val] = 1;
            } else {
                core_occu[raw_physical_id_.back()][val] += 1;
            }

            int cur_ht_id = core_occu[raw_physical_id_.back()][val] - 1;  // from 0 to HT_COUNT_PER_CORE-1

            if (threads_by_socket_ht_[raw_physical_id_.back()].size() <= cur_ht_id) {
                threads_by_socket_ht_[raw_physical_id_.back()].push_back(vector<int>());
            }

            threads_by_socket_ht_[raw_physical_id_.back()][cur_ht_id].push_back(raw_processor_.back());
        }

        if (strstr(buffer,"cpu cores") != nullptr) {
            raw_cpu_cores_.push_back(val); cpu_cores_val = val;
        }
    }

    fclose(fp);

    //check avail
    for (auto _val : raw_siblings_)
        assert(_val = siblings_val);
    for (auto _val : raw_cpu_cores_)
        assert(_val = cpu_cores_val);

    //get constant
    total_thread_count_ = raw_processor_.size();
    thread_per_socket_ = siblings_val;
    thread_per_core_ = siblings_val / cpu_cores_val;
    total_core_count_ = total_thread_count_ / thread_per_core_;
    total_socket_count_ = socket_proc_count_map_.size();
    core_per_socket_ = total_core_count_ / total_socket_count_;

    //socket mapping can be directly read from physical_id
    socket_id_by_thread_ = raw_physical_id_;

    //construct the map of printed core_id to continual core_id starts from 0
    for (auto core_id : raw_core_id_)
        core_id_map_[core_id] = core_id;

    // assume that the core id layout of different sockets is the same
    int act_local_core_id = 0;
    for (auto & kv : core_id_map_) {
        kv.second = act_local_core_id;
        act_local_core_id++;
    }

    //construct continual_core_id_in_socket_by_thread_ according the core_id_map_
    for (auto core_id : raw_core_id_)
        continual_core_id_in_socket_by_thread_.push_back(core_id_map_[core_id]);

    //construct continual_global_core_id_by_thread_
    for (int i = 0; i < continual_core_id_in_socket_by_thread_.size(); i++) {
        continual_global_core_id_by_thread_.push_back(socket_id_by_thread_[i] * core_per_socket_ + continual_core_id_in_socket_by_thread_[i]);
    }

    threads_by_continual_global_core_id_.resize(total_core_count_);
    for (int i = 0; i < total_thread_count_; i++) {
        int global_core_id = continual_global_core_id_by_thread_[i];
        threads_by_continual_global_core_id_[global_core_id].push_back(i);
    }
}