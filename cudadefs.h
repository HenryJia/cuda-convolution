#define CC 52
#define BLOCK_DIM1 1024
#define BLOCK_DIM2 32

#ifndef IDX2C // For Column Major
#define IDX2C(i,j,ld) (j * ld + i) // i is column, j is row, ld is total number of columns
#endif

#define O_TILE_DIM(block, filter) (block - filter + 1)
