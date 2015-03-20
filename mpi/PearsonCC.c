#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Declaration of global variables
const int MAX_LENGTH = 1000000;
int comm_sz; 
int rank;
clock_t begin_init, begin_calc, end;
double time_serial_from_init, time_serial_from_calc, time_parallel_from_init, time_parallel_from_calc;

// Method declarations
void serialPCC();
void parallelPCC();

int main(void) {
    
    // Initialising MPI, getting comm size and each process rank
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Executing the serial method.
    // If statement to make sure only 1 process executes the serial method (process with rank 0).
    if (rank == 0) {
        serialPCC();
    }
        
    // Executing the parallel method.
    // If statement to make sure it is only executed when > 1 process exists.
    if (comm_sz > 1) {
        parallelPCC();
        
        // If parallel executes correctly, printing the speedup achieved
        if (rank == 0) {
            printf("\nSPEEDUP ACHIEVED BY PARALLELIZATION (Calculation Only): %.15f ms", (time_serial_from_calc - time_parallel_from_calc));
            printf("\nSPEEDUP ACHIEVED BY PARALLELIZATION (Including Array Initialization): %.15f ms", (time_serial_from_init - time_parallel_from_init));
            printf("\nPERCENTAGE SPEEDUP ACHIEVED: %.0f%%\n\n", (1-(time_parallel_from_calc/time_serial_from_calc))*100);
        }
    }
    else {
        printf("\nERROR. Cannot carry out parallel implementation since you chose to run less than 2 processes.\n\n");
    }
    
    MPI_Finalize();
    return 0;
}


// Serial Calculation Method
void serialPCC() {
    printf("\n===================== START SERIAL =====================\n");
    
    // Store the starting clock time
    begin_init = clock();
    
    // Declaration of input arrays and other variables
    double* a = malloc((MAX_LENGTH) * sizeof(double));
    double* b = malloc((MAX_LENGTH) * sizeof(double));
    double totalA = 0;
    double totalB = 0;
    double tempA = 0;
    double tempB = 0;
    double tempSum = 0;
    
    /* 
     * This loop initializes each space in the arrays and as it iterates also sums the
     * total of all values in a (used to calculate the mean).
     * Since a[i+1] equals b[i], no need to recalculate a[i] except for a[0]. This saves
     * repeated calculations being made.
     */
    for (int i = 0; i < MAX_LENGTH; i++) {
        
        if (i == 0) {
            a[i] = sin(i);
        }
        
        b[i] = sin(i+1);
        
        if (i < MAX_LENGTH-1) {
            a[i+1] = b[i];
        }
            
        totalA += a[i];
    }
    
    totalB = totalA + b[MAX_LENGTH-1] - a[0];
    
    // Store the starting calculation clock time
    begin_calc = clock();
    
    // Calculating the printing the means.
    double meanA = totalA / MAX_LENGTH;
    double meanB = totalB / MAX_LENGTH;
    printf("\nMean A: %.12f", meanA);
    printf("\nMean B: %.12f\n", meanB);
        
    /* 
     * This loop calculates the numerator in both the standard deviation and pearson
     * correlation coefficient calcualtions.
     */
    for (int j = 0; j < MAX_LENGTH; j++) {
        tempA += (a[j] - meanA) * (a[j] - meanA);
        tempB += (b[j] - meanB) * (b[j] - meanB);
        
        // Pearson
        tempSum += (a[j] - meanA) * (b[j] - meanB);
    }
    
    // Calculate standard deviations
    double sdA = sqrt(tempA / MAX_LENGTH);
    double sdB = sqrt(tempB / MAX_LENGTH);
    printf("\nStandard Deviation A: %.12f", sdA);
    printf("\nStandard Deviation B: %.12f\n", sdB);
    
    // Calculate pearson correlation coefficient
    tempSum = tempSum / MAX_LENGTH;
    double pcc = tempSum / (sdA * sdB);
    printf("\nPearson Correlation Coefficient: %.12f\n", pcc);
    
    // Take ending clock time and print time taken
    end = clock();
    time_serial_from_calc = ((double)(end - begin_calc) / CLOCKS_PER_SEC)*1000;
    printf("\nTime Taken (Calculation Only): %.3f ms", time_serial_from_calc);
    time_serial_from_init = ((double)(end - begin_init) / CLOCKS_PER_SEC)*1000;
    printf("\nTime Taken (Including Array Initialization): %.3f ms\n", time_serial_from_init);
    
    // Release the memory of the arrays
    free(a);
    free(b);
    
    printf("\n====================== END SERIAL ======================\n");
}


// Parallel Calculation Method
void parallelPCC() {
    
    // Store the starting clock time
    if (rank == 0) {
        printf("===================== START PARALLEL =====================\n");
        
        begin_init = clock();
    }
    
    // Declaration of global input arrays and other variables.
    double* a = malloc((MAX_LENGTH) * sizeof(double));
    double* b = malloc((MAX_LENGTH) * sizeof(double));
    double tempSumA = 0;
    double tempSumB = 0;
    double totalA = 0;
    double totalB = 0;
    double meanA = 0;
    double meanB = 0;
    double lastB = 0;

    /*
     * If MAX_LENGTH is not divisible by the number of processes, then some
     * processes will need to do more work than other.
     * Taking this into account, the following calculates the size of each
     * processes local array.
     */
    int localSize = 0;
    int remainder = MAX_LENGTH % comm_sz;
    if (remainder == 0) {
        localSize = MAX_LENGTH/comm_sz;
    }
    else {
        localSize = MAX_LENGTH/comm_sz;
        if (rank < remainder) {
            localSize++;
        }
    }
    
    // Declaring the local process arrays
    double* local_a = malloc(localSize * sizeof(double));
    double* local_b = malloc(localSize * sizeof(double));
    
    /*
     * Since each process will start iterating at i = 0, the following calculates
     * an offset from 0 for each process otherwise they would all calculate the same thing.
     * This also takes into account again that MAX_LENGTH may not be divisible by the
     * number of processes.
     */
    double offset = 0;
    if (rank < remainder) {
        offset = rank * localSize;
    }
    else {
        offset = (rank * localSize) + remainder;
    }
    
    // Each local array is then initialized and the total of local a
    // summed, just as in the serial method above.
    for (int i = 0; i < localSize; i++) {
        
        if (i == 0) {
            local_a[i] = sin(i + offset);
        }
        
        local_b[i] = sin(i + 1 + offset);
        
        if (i < localSize-1) {
            local_a[i+1] = local_b[i];
        }
    
        tempSumA += local_a[i];
    }
    
    // b[MAXLENGTH-1] must be sent back to process 0 so that it can calculate
    // the total of array b (since 'totalB = totalA + b[MAXLENGTH-1]')
    if (rank == comm_sz-1) {
        lastB = local_b[localSize-1];
        MPI_Send(&lastB, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Reducing all the local temproary summations into 1 overall value for totalA.
    MPI_Reduce(&tempSumA, &totalA, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
    // Store the starting calculation clock time
    begin_calc = clock();
    
    // Process 0 now receives b[MAXLENGTH-1] and calculates the means
    if (rank == 0) {
        
        MPI_Recv(&lastB, 1, MPI_DOUBLE, comm_sz-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
        meanA = totalA / MAX_LENGTH;
        
        totalB = totalA + lastB;
        meanB = totalB / MAX_LENGTH;
        
        printf("\nMean A: %.12f", meanA);
        printf("\nMean B: %.12f\n", meanB);
    }
        
    // Broadcasting the calculated means back to all other processes
    MPI_Bcast(&meanA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meanB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
    
    // Declaring variables needed to calculate standard deviation and pearson correlation coefficient.
    double tempA = 0;
    double tempB = 0;
    double tempSum = 0;
    double totalA2 = 0;
    double totalB2 = 0;
    double totalTempSum = 0;
    
    /* 
     * This loop calculates the numerator in both the standard deviation and pearson
     * correlation coefficient calcualtions.
     * Works in exactly same way as loop in serial method, except each process only
     * does calculations for it local array.
     */
    for (int j = 0; j < localSize; j++) {
        
        tempA += (local_a[j] - meanA) * (local_a[j] - meanA);
        tempB += (local_b[j] - meanB) * (local_b[j] - meanB);

        // Pearson
        tempSum += (local_a[j] - meanA) * (local_b[j] - meanB);
    }
    
    // Locally calculated values from each process are then reduced to an overall sum
    MPI_Reduce(&tempA, &totalA2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tempB, &totalB2, 1, MPI_DOUBLE, MPI_SUM, 1, MPI_COMM_WORLD);
    MPI_Reduce(&tempSum, &totalTempSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Releasing the memory of the arrays
    free(local_a);
    free(local_b);
    free(a);
    free(b);
    
    // Process 1 calculates the standard deviation of B and then sends this
    // value to process 0.
    // Process 0 calculates the standard deviation of A.
    double sdB = 0;
    if (rank == 1) {
        sdB = sqrt(totalB2 / MAX_LENGTH);
        MPI_Send(&sdB, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        double sdA = sqrt(totalA2 / MAX_LENGTH);
        printf("\nStandard Deviation A: %.12f", sdA);
        
        MPI_Recv(&sdB, 1, MPI_DOUBLE, 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\nStandard Deviation B: %.12f\n", sdB);
        
        // Calculating the pearson correlation coefficient
        totalTempSum = totalTempSum / MAX_LENGTH;
        double pcc = totalTempSum / (sdA * sdB);
        printf("\nPearson Correlation Coefficient: %.12f\n", pcc);

        // Take ending clock time and print time taken
        end = clock();
        time_parallel_from_calc = ((double)(end - begin_calc) / CLOCKS_PER_SEC)*1000;
        printf("\nTime Taken (Calculation Only): %.3f ms", time_parallel_from_calc);
        time_parallel_from_init = ((double)(end - begin_init) / CLOCKS_PER_SEC)*1000;
        printf("\nTime Taken (Including Array Initialization): %.3f ms\n", time_parallel_from_init);

        printf("\n====================== END PARALLEL ======================\n");
    }
    
}