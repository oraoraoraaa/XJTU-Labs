/* Test harness for the QL qsort program.
 *
 * The QL program declares a global `int a[10]` and a top-level statement
 * `qsort(a[], 0, 9,)`.  The compiled assembly exposes:
 *      ql_a        -- the 10-element global array (8-byte / long long ints)
 *      ql_entry    -- the QL top-level code (calls qsort on ql_a)
 *
 * This driver fills ql_a with an unsorted sequence, invokes the QL entry
 * point, prints the result, and checks that the array came out sorted.
 *
 * Symbol names assemble on both Linux/Kunpeng (no leading underscore) and
 * macOS (leading underscore) because codegen emits both label forms.
 */
#include <stdio.h>

extern long long ql_a[10];
extern void ql_entry(void);

static void print_array(const char *tag) {
    printf("%s", tag);
    for (int k = 0; k < 10; k++) printf(" %lld", ql_a[k]);
    printf("\n");
}

int main(void) {
    long long init[10] = {5, 3, 8, 1, 9, 2, 7, 4, 6, 0};
    for (int k = 0; k < 10; k++) ql_a[k] = init[k];

    print_array("before:");
    ql_entry();              /* runs the QL program: qsort(a, 0, 9) */
    print_array("after :");

    for (int k = 1; k < 10; k++) {
        if (ql_a[k - 1] > ql_a[k]) {
            printf("FAIL: array is not sorted\n");
            return 1;
        }
    }
    printf("OK: array is sorted\n");
    return 0;
}
