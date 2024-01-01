#include "basic_definition.hpp"

#include <signal.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <string>

using namespace std;

string strprintf(const char *fmt, ...) {
    constexpr size_t BUFF_SZ = 10240;
    static __thread char buf[BUFF_SZ];
    va_list ap;
    va_start (ap, fmt);

    vsnprintf(&buf[0], BUFF_SZ, fmt, ap);

    string ret = string(buf);

    va_end (ap);

    return ret;
}

void exit9() {
    kill(getpid(), 9);
}
