#define TS 32

typedef float value_type;

// First naive implementation
__kernel void mul0(int M, int N, int K,
                       const __global value_type* A,
                       const __global value_type* B,
                       __global value_type* C)
{
    // Thread identifiers
    auto global_row = get_global_id(0); // row ID of C (0..M)
    auto global_col = get_global_id(1); // col ID of C (0..N)

    // Compute a single element (loop over K)
    value_type acc = 0;
    for (auto k = 0; k < K; k++) {
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

