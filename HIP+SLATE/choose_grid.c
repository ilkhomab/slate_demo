#include <math.h>
#include <stdint.h>

void choose_grid_f(int nprocs, int* nprow, int* npcol)
{
    int p = (int)floor(sqrt((double)nprocs));
    while (p > 1 && (nprocs % p) != 0) p--;
    *nprow = p;
    *npcol = nprocs / p;
}

