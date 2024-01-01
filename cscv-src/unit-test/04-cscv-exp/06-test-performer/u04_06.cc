#include "test/test_performer.hpp"

#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    // set omp thread stack size
    Test_performer tester(argc, argv);

    tester.run();

    return 0;
}
