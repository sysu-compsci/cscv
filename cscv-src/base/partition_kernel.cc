#include "partition_kernel.hpp"

#include <cmath>

using namespace std;

/**
 * return value:
 *  -1.0: invalid input, if (first < 1 || second < 1)
 *  return first / second
 *  1.0: balance, first == second
 */
template <class T1, class T2>
inline double get_balance_2d(T1 first, T2 second) {
    if (first < 1 || second < 1)
        return -1.0;

    return first * 1.0 / second;
}

/**
 * A valid return value should be in (0.0, 1.0]
 * 1.0: perfect
 * 0.0: extremely bad
 * -1.0: Invalid
 **/
template <class T1, class T2>
inline double get_balance_ratio_distance(T1 first, T2 second, double prefer_ratio) {
    double balance_2d = get_balance_2d(first, second);
    if (prefer_ratio <= 0.0 && balance_2d <= 0.0)
        return -1.0;
    return min(balance_2d, prefer_ratio) / max(balance_2d, prefer_ratio);
}

pair<int, int> checker_2d_dim_create(const int n, const double prefer_ratio) {
    pair<int, int> best_result = make_pair(1, n);
    for (int i = 1; i <= sqrt(n); i++) {
        if (n % i == 0) {
            if (get_balance_ratio_distance(best_result.first, best_result.second, prefer_ratio) <
                get_balance_ratio_distance(i, n / i, prefer_ratio))
                best_result = make_pair(i, n / i);
            if (get_balance_ratio_distance(best_result.first, best_result.second, prefer_ratio) <
                get_balance_ratio_distance(n / i, i, prefer_ratio))
                best_result = make_pair(n / i, i);
        }
    }

    // PRINTF("BR: %d %d\n", best_result.first, best_result.second);
    return best_result;
}
