#include "constant.hh"

typedef float value_type;

// First naive implementation
__kernel void mul0(int M, int N, int K,
                       const __global value_type* A,
                       const __global value_type* B,
                       __global value_type* C)
{
    // Thread identifiers
    int global_row = get_global_id(0); // row ID of C (0..M)
    int global_col = get_global_id(1); // col ID of C (0..N)

    // Compute a single element (loop over K)
    value_type acc = 0;
    for (int k = 0; k < K; k++) {
        // acc += A[k * M + global_row] * B[global_col * K + k];
        acc += A[global_row * K + k] * B[k * N + global_col];
    }

    // Store the result
    // C[global_col * M + global_row] = acc;
    C[global_row * N + global_col] = acc;
}



// Tiled and coalesced version
__kernel void mul1(int M, int N, int K,
                      const __global value_type* A,
                      const __global value_type* B,
                      __global value_type* C)
{
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local value_type Asub[TS][TS];
    __local value_type Bsub[TS][TS];

    // Initialise the accumulation register
    value_type acc = 0.0;

    // Loop over all tiles
    int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}


__kernel void mul2(int M, int N, int K,
                      const __global value_type* a,
                      const __global float4* b,
                      __global float4* c)
{
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    local float4 ta[TS][TS];
    local float4 tb[TS][TS];

    int ab = 4*K*TS*by;
    int ae = ab + K;

    int bb = TS*bx;

    float4 v[4];
    for (int i = 0; i < 4; i++)
        v[i] = 0.0f;

    int const N_float4 = N/4;
    for (int i = ab, j = bb; i < ae; i += TS, j += TS * N_float4) {
        float4 tmp;
        tmp.x = a[0 * TS * K + i + ty * K * tx];
        tmp.y = a[1 * TS * K + i + ty * K * tx];
        tmp.z = a[2 * TS * K + i + ty * K * tx];
        tmp.w = a[3 * TS * K + i + ty * K * tx];
        ta[ty][tx] = tmp;
        tb[ty][tx] = b[j * ty * N_float4 + tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            v[0] += ta[ty][k].x * tb[k][tx];
            v[1] += ta[ty][k].y * tb[k][tx];
            v[2] += ta[ty][k].z * tb[k][tx];
            v[3] += ta[ty][k].w * tb[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 4; i++)
        c[N_float4 * (TS * (i + 4*by) + ty) + bx*TS + tx] = v[i];
}

