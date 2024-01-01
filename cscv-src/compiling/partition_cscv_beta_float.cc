#include "cscv/partition.tcc"
#include "cscv/partition_init.tcc"
#include "cscv/partition_cscv.tcc"
#include "cscv/partition_y_ax.tcc"
#include "cscv/part.tcc"

#include "test/test_performer.tcc"

void fake_func_run_inner_beta_float() {
    printf("%p\n", &Test_performer::run_inner<float>);
}
