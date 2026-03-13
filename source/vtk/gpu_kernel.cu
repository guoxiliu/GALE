#include "gpu_kernel.h"

//#define PRINT_ON 1

__global__ void critPoints(const int * __restrict__ VV, // formerly int
                           const int * __restrict__ VV_index, // formerly int
                           const int * __restrict__ VV_to_global,
                           int * __restrict__ valences,
                           const int points,
                           const int max_VV_guess, // formerly SFCP_ColumnNum
                           const double * __restrict__ scalar_values,
                           const int * __restrict__ partition, // NEW
                           const vtkIdType * __restrict__ vvi, // nth -> global
                           const vtkIdType * __restrict__ ivvi, // dense global -> nth
                           unsigned int * __restrict__ classes) {
    extern __shared__ unsigned int block_shared[];

    unsigned int *component_edits = &block_shared[0],
                 *upper_edits = &block_shared[0],
                 *lower_edits = &block_shared[1];
    vtkIdType *neighborhood = (vtkIdType*)(block_shared+2); // NEW: Multiple datatype shared

    // Set execution mask to EXCLUDE no work / out-of-partition points
    bool execution_mask = true;
    /*
        1) Parallelize VV on second-dimension

           my_1d and my_2d have the GLOBAL vertex IDs
           indirect_my*    have the LOCAL vertex IDs used to index in all
                            arrays except ivvi
    */
    //const int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tid = (max_VV_guess * 11) + threadIdx.x;
    const vtkIdType indirect_my_1d = tid / max_VV_guess,
                    my_1d = vvi[indirect_my_1d];
    vtkIdType my_2d = VV[tid], // NOTE: Requires deref via VV_to_global IFF VV[tid] != -1
              indirect_my_2d;
    if (my_2d < 0) {
        execution_mask = false;
    }
    else {
        my_2d = VV_to_global[my_2d];
        indirect_my_2d = ivvi[my_2d];
    }

    if (VV_index[indirect_my_1d] <= 0 || partition[indirect_my_1d] == 0) {
        /*
        printf("Block %d Thread %02d exits due to partitioning or OOB data\n",
               blockIdx.x, threadIdx.x);
        */
        execution_mask = false;
    }
    // Prefix scan as anti-duplication
    for (vtkIdType i = indirect_my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] < 0) {
            continue;
        }
        if (VV_to_global[VV[i]] == my_2d) {
            /*
            printf("Block %d Thread %02d exits from prefix scan\n",
                   blockIdx.x, threadIdx.x);
            */
            execution_mask = false;
        }
    }

    /*
        2) Read the scalar value used for point classification and classify
           your 2nd point relative to your 1st scalar value as upper (-1) or
           lower (1) neighbor

           For parity with Paraview, tied scalar values are tie-broken such
           that the lesser vertex ID is assigned as lower
    */

    /* These variables COULD be const, but non-executing threads cannot be
       allowed to potentially read bad memory addresses */
    int my_class;
    vtkIdType min_my_1d, max_my_1d;
    if (execution_mask) {
        // GALE needs to use GLOBAL scalar values
        my_class = 1 - (
                        (scalar_values[my_2d] == scalar_values[my_1d] ?
                            my_2d < my_1d :
                            scalar_values[my_2d] < scalar_values[my_1d])
                        << 1);
        // Allow other threads in the block to read your class for matching
        // Not SHMEM yet -- could be as there are no cross-block reads here
        valences[tid] = my_class;
        min_my_1d = (indirect_my_1d * max_VV_guess);
        max_my_1d = min_my_1d + VV_index[indirect_my_1d];
        // Initialize your neighborhood with your own 2d value
        // If you END the union-find without editing your neighborhood, you are
        // the root of a component of your class.
        neighborhood[threadIdx.x] = my_2d;
    }
    __syncthreads();
    if (execution_mask) {
        #ifdef PRINT_ON
        printf("Block %d Thread %02d init on my_1d %ld and my_2d %ld with "
               "valence %d based on %lf (block) vs %lf (thread)\n",
               blockIdx.x, threadIdx.x, my_1d, my_2d, my_class,
               scalar_values[my_1d], scalar_values[my_2d]);
        #endif
    }
    /*
        3) Union-Find: All threads with a common classification attempt to
            union such that the lowest-connected point with the same class
            is set as your neighborhood value (accelerating future decisions)

            Repeat until convergence. The `to_converge` variable should NEVER
            be a termination condition but guarantees kernel termination (even
            if all points share the same class, Union-Find CANNOT execute more
            iterations than this maximum).
    */
    bool upper_converge = false, lower_converge = false;
    int to_converge = 0;

    while (!(upper_converge && lower_converge) && to_converge < max_VV_guess) {
        // Reset convergence indicators for the iteration
        if (threadIdx.x == 0) {
            if (execution_mask) {
                /*
                printf("Block %d Iteration %d: upper_edits %u lower_edits %u "
                       "Neighborhood %ld %ld %ld %ld %ld %ld %ld %ld "
                                    "%ld %ld %ld %ld %ld %ld %ld %ld "
                                    "%ld %ld %ld %ld %ld %ld %ld %ld "
                                    "%ld %ld %ld %ld %ld %ld %ld %ld\n",
                        blockIdx.x, to_converge, *upper_edits, *lower_edits,
                        neighborhood[0],neighborhood[1],neighborhood[2],neighborhood[3],
                        neighborhood[4],neighborhood[5],neighborhood[6],neighborhood[7],
                        neighborhood[8],neighborhood[9],neighborhood[10],neighborhood[11],
                        neighborhood[12],neighborhood[13],neighborhood[14],neighborhood[15],
                        neighborhood[16],neighborhood[17],neighborhood[18],neighborhood[19],
                        neighborhood[20],neighborhood[21],neighborhood[22],neighborhood[23],
                        neighborhood[24],neighborhood[25],neighborhood[26],neighborhood[27],
                        neighborhood[28],neighborhood[29],neighborhood[30],neighborhood[31]);
                */
            }
            *upper_edits = 0;
            *lower_edits = 0;
        }
        to_converge++;
        __syncthreads();

        if (execution_mask) {
            /*
            printf("Block %d thread %02d clears START UF sync %d\n", blockIdx.x,
                    threadIdx.x, to_converge);
            */

            // Union-Find iteration
            for(vtkIdType i = min_my_1d; i < max_my_1d; i++) {
                // Find a matching valence
                const vtkIdType indirect_candidate_component_2d = VV[i],
                                candidate_component_2d = VV_to_global[indirect_candidate_component_2d];
                /*
                    candidate_component_2d = VV[i], // Within bounds, should be >= 0
                                indirect_candidate_component_2d = VV_to_global[candidate_component_2d];
                */
                if (i != tid && valences[i] == my_class) {
                    const vtkIdType start_2d = indirect_candidate_component_2d*max_VV_guess,
                                    stop_2d  = start_2d + VV_index[indirect_candidate_component_2d];
                    #ifdef PRINT_ON
                    // Pure debug for this problem
                    if (my_2d > 2000) {
                        printf("Block %d Thread %02d has matching candidate valence for point %ld, which was looked up indirectly as %ld with iteration range %ld->%ld\n",
                                blockIdx.x, threadIdx.x, candidate_component_2d,
                                indirect_candidate_component_2d,
                                start_2d, stop_2d);
                    }
                    #endif
                    if (start_2d > blockDim.x * gridDim.x) {
                        /*
                        printf("Block %d Thread %02d has bad iterator from %ld "
                                "to %ld based on direct %ld and indirect %ld "
                                "at iteration %ld\n",
                                blockIdx.x, threadIdx.x, start_2d, stop_2d,
                                candidate_component_2d,
                                indirect_candidate_component_2d, i);
                        */
                    }
                    /* Do you see your 2d in their 2d? If so, update your
                       neighborhood with the minimum of your two points
                    */
                    for(vtkIdType j = start_2d; j < stop_2d; j++) {
                        const vtkIdType theirNeighborhood = neighborhood[i-min_my_1d];
                        #ifdef PRINT_ON
                        if (my_2d > 2000) {
                            printf("Block %d Thread %02d (owner of %ld) checking %ld's %ld'th point in scan (VV[j]=%d, VV2G[VV[j]]=%d) neighborhood=%ld vs my neighborhood=%ld\n",
                                    blockIdx.x,
                                    threadIdx.x,
                                    my_2d,
                                    candidate_component_2d,
                                    j-start_2d,
                                    VV[j],
                                    VV_to_global[VV[j]],
                                    theirNeighborhood,
                                    neighborhood[threadIdx.x]);
                        }
                        #endif
                        // Within bounds, VV[j] >= 0
                        if (VV_to_global[VV[j]] == my_2d && neighborhood[threadIdx.x] > theirNeighborhood) {
                            #ifdef PRINT_ON
                            printf("Block %d Thread %02d neighborhood update "
                                    "iteration %0d moves %ld --> %ld\n",
                                    blockIdx.x, threadIdx.x, to_converge,
                                    neighborhood[threadIdx.x],
                                    theirNeighborhood);
                            #endif
                            neighborhood[threadIdx.x] = theirNeighborhood;
                            // Denote LACK of convergence this iteration
                            atomicAdd(component_edits+((my_class+1)/2), 1);
                        }
                    }
                }
            }
        }
        // End Union-Find iteration

        __syncthreads();
        upper_converge = (*upper_edits == 0);
        lower_converge = (*lower_edits == 0);
        if (execution_mask) {
            #ifdef PRINT_ON
            printf("Block %d Thread %02d ends UF loop %02d with upper_edits %u "
                    "lower_edits %u (loop condition: %d = [%d[!%d && %d] && %d])\n",
                    blockIdx.x, threadIdx.x, to_converge, *upper_edits, *lower_edits,
                    !(upper_converge && lower_converge) && to_converge < max_VV_guess,
                    !(upper_converge && lower_converge),
                    upper_converge, lower_converge,
                    to_converge < max_VV_guess
                    );
            #endif
        }
    }
    // End all Union-Finds

    /*
       4) Each root of each component increments the counter for its class to
            contribute to the final classification of the per-block point
    */
    if (execution_mask && neighborhood[threadIdx.x] == my_2d) {
        const vtkIdType memoffset = ((indirect_my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        #ifdef PRINT_ON
        printf("Block %d Thread %02d Writes %s component @ mem offset %ld "
                "(old: %u) after %02d iterations\n",
                blockIdx.x, threadIdx.x,
                (my_class == -1 ? "Upper" : "Lower"), memoffset, old, to_converge);
        #endif
    }
    __syncthreads();

    /*
        5) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    if (threadIdx.x == 0) {
        const vtkIdType my_classes = indirect_my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        if (execution_mask) {
            #ifdef PRINT_ON
            printf("Block %d Thread %02d Verdict: my_1d (%ld) has %u upper and %u "
                    "lower after %02d iterations\n",
                    blockIdx.x, threadIdx.x, my_1d, upper, lower, to_converge);
            #endif
        }
        // Set classification
        if (upper >= 1 && lower == 0) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "MINIMUM");
            #endif
            classes[my_classes+2] = MINIMUM_CLASS;
        }
        else if (upper == 0 && lower >= 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "MAXIMUM");
            #endif
            classes[my_classes+2] = MAXIMUM_CLASS;
        }
        else if (upper == 1 && lower == 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "REGULAR");
            #endif
            classes[my_classes+2] = REGULAR_CLASS;
        }
        else {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "SADDLE");
            #endif
            classes[my_classes+2] = SADDLE_CLASS;
        }
    }
    // END OF KERNEL
    /*
    printf("Block %d Thread %02d exits normally at end of computation with "
            "%02d iterations\n",
            blockIdx.x, threadIdx.x, to_converge);
    */
}


