/*
 * Compile using: gcc -O3 --std=c23 -fopenmp main.c
 * Set EXPORT OMP_NUM_THREADS=4 before running
 */

#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <assert.h>

/*
 * Instead of computing the entire decimal expansion of 2^n, we work with just the last 40 digits.
 * Essentially, we are looking at the number 2^n mod 10^40.
 * If this number already contains odd digits, we know that 2^n has odd digits.
 * However, after checking around 1.33 * 10^11 numbers, a first number is found where the 40 last digits are even.
 * So DIGITS must be increased to continue the search with this approach.
 */
int DIGITS = 40;
// we handle batches of 1 billion numbers at a time
long long BATCH = 1'000'000'000;

/*
 * Multipy the decimal number whose digits are stored in tail by 16.
 * The number is stored in little-endian order.
 *
 * Multiplying by 16 is the same as multiplying by 10 and 6, then adding the results.
 * Multiplication by 10 is easy, just shift all digits one place to the left.
 * We can do both in one pass, by adapting the carry while multiplying by 6.
 */
inline void times16(int *tail) {
    int carry = 0, next_carry;
    // good signifies whether the number has only even digits
    int good = 1;
    for (int i = 0; i < DIGITS; i++) {
        next_carry = tail[i];
        tail[i] *= 6;
        tail[i] += carry;
        next_carry += tail[i] / 10;
        tail[i] %= 10;
        // use &= to avoid branching
        good &= tail[i] % 2 == 0;
        carry = next_carry;
    }
    if (good) {
        // print the number if it has only even digits
        for (int j = DIGITS - 1; j >= 0; j--) {
            putchar(tail[j] + '0');
        }
        putchar('\n');
    }
}

int main() {
    // I ran this on a 4-core machine, it could be parallelized further by multiplying at each step with 2^CORES
    assert(omp_get_max_threads() == 4);
    #pragma omp parallel
    {
        int tail[DIGITS];
        memset(tail, 0, sizeof(tail));
        tail[0] = 1;
        // the threads start with 1, 2, 4, 8 respectively
        for (int i = 0; i < omp_get_thread_num(); i++) {
            tail[0] *= 2;
        }
        long long steps = 0;
        while (1) {
            // use batched inner loop to avoid branching for the prints
            for (int i = 0; i < BATCH; i++) {
                // each thread is skipping over 4 numbers at a time, hence we multiply by 16 = 2^4
                times16(tail);
                steps++;
            }
            printf("Steps: %lld from %d\n", steps, omp_get_thread_num());
        }
    }
}