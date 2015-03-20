#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Declaration of global variables
const int MAX_LENGTH = 1000000;
double begin_init, begin_calc, end;
double time_serial_from_init, time_serial_from_calc, time_parallel_from_init, time_parallel_from_calc;

// Method declarations
void serialPCC();
void parallelPCC();

int main(void) {

    // Executing the serial method
    serialPCC();

    // Executing the parallel method
    parallelPCC();

    // Printing speedup
    printf("\nSPEEDUP ACHIEVED BY PARALLELIZATION (Calculation Only): %.3f ms", (time_serial_from_calc - time_parallel_from_calc));
    printf("\nPERCENTAGE SPEEDUP ACHIEVED (Calculation Only): %.0f%%\n", (1-(time_parallel_from_calc/time_serial_from_calc))*100);
    printf("\nSPEEDUP ACHIEVED BY PARALLELIZATION (Including Array Initialization): %.3f ms", (time_serial_from_init - time_parallel_from_init));
    printf("\nPERCENTAGE SPEEDUP ACHIEVED (Including Array Initialization): %.0f%%\n\n", (1-(time_parallel_from_init/time_serial_from_init))*100);

    return 0;
}


// Serial Calculation Method
void serialPCC() {
    printf("\n===================== START SERIAL =====================\n");

    // Store the starting clock time (before array initialization)
    begin_init = omp_get_wtime();

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
     * Since a[i+1] equals b[i], no need to calculate a[i] except for a[0]. This saves
     * nearly MAX_LENGTH repeated calculations being made.
     */
    a[0] = sin(0);
    totalA += a[0];
    for (int i = 0; i < MAX_LENGTH-1; i++) {

        b[i] = sin(i+1);
        a[i+1] = b[i];

        totalA += a[i+1];
    }
    b[MAX_LENGTH-1] = sin(MAX_LENGTH);

    /*
     * Since array 'b' is essentially array 'a' shifted one place left, total of array 'b' is
     * equal to array 'a' plus the last position in array 'b' (b[MAX_LENGTH-1]) minus a[0].
     */
    totalB = totalA + b[MAX_LENGTH-1] - a[0];

    // Store the starting calculation clock time
    begin_calc = omp_get_wtime();

    // Calculating and printing the means.
    double meanA = totalA / MAX_LENGTH;
    double meanB = totalB / MAX_LENGTH;
    printf("\nMean A: %.12f", meanA);
    printf("\nMean B: %.12f\n", meanB);

    /*
     * This loop calculates the numerator in both the standard deviation and pearson
     * correlation coefficient calculations.
     */
    for (int j = 0; j < MAX_LENGTH; j++) {
        // Standard Deviation
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
    end = omp_get_wtime();
    time_serial_from_calc = (double)(end - begin_calc)*1000;
    printf("\nTime Taken (Calculation Only): %.3f ms", time_serial_from_calc);
    time_serial_from_init = (double)(end - begin_init)*1000;
    printf("\nTime Taken (Including Array Initialization): %.3f ms\n", time_serial_from_init);

    // Release the memory of the arrays
    free(a);
    free(b);

    printf("\n====================== END SERIAL ======================");
}


// Parallel Calculation Method
void parallelPCC() {
    printf("\n===================== START PARALLEL =====================\n");

    // Store the starting clock time (before array initialization)
    begin_init = omp_get_wtime();

    // Declaration of input arrays and other variables
    double* a = malloc((MAX_LENGTH) * sizeof(double));
    double* b = malloc((MAX_LENGTH) * sizeof(double));
    double totalA = 0;
    double totalB = 0;
    double meanA = 0;
    double meanB = 0;
    double tempA = 0;
    double tempB = 0;
    double tempSum = 0;
    double sdA = 0;
    double sdB = 0;

    /*
     * This loop initializes each space in the arrays and as it iterates also sums the
     * total of all values in a (used to calculate the mean).
     * Since a[i+1] equals b[i], no need to calculate a[i] except for a[0]. This saves
     * nearly MAX_LENGTH repeated calculations being made.
     */
    a[0] = sin(0);
    totalA += a[0];
    #pragma omp parallel for reduction(+:totalA)
    for (int i = 0; i < MAX_LENGTH-1; i++) {

        b[i] = sin(i+1);
        a[i+1] = b[i];

        totalA += b[i];
    }
    b[MAX_LENGTH-1] = sin(MAX_LENGTH);

    /*
     * Since array 'b' is essentially array 'a' shifted one place left, total of array 'b' is
     * equal to array 'a' plus the last position in array 'b' (b[MAX_LENGTH-1]) minus a[0].
     */
    totalB = totalA + b[MAX_LENGTH-1] - a[0];

    // Store the starting calculation clock time
    begin_calc = omp_get_wtime();

    // Calculating and printing the means.
    // 2 threads calculate each mean simultaneously.
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            meanA = totalA / MAX_LENGTH;
            printf("\nMean A: %.12f", meanA);
        }

        #pragma omp section
        {
            meanB = totalB / MAX_LENGTH;
            printf("\nMean B: %.12f\n", meanB);
        }
    }


    /*
     * This loop calculates the numerator in both the standard deviation and pearson
     * correlation coefficient calcualtions.
     */
    #pragma omp parallel for reduction(+:tempA) reduction(+:tempB) reduction(+:tempSum)
    for (int j = 0; j < MAX_LENGTH; j++) {
        // Standard Deviation
        tempA += (a[j] - meanA) * (a[j] - meanA);
        tempB += (b[j] - meanB) * (b[j] - meanB);

        // Pearson
        tempSum += (a[j] - meanA) * (b[j] - meanB);
    }

    // Calculating standard deviations
    // Again, 2 threads used to calculate sdA & sdB simultaneously
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            sdA = sqrt(tempA / MAX_LENGTH);
            printf("\nStandard Deviation A: %.12f", sdA);
        }

        #pragma omp section
        {
            sdB = sqrt(tempB / MAX_LENGTH);
            printf("\nStandard Deviation B: %.12f\n", sdB);
        }
    }

    // Calculate pearson correlation coefficient
    tempSum = tempSum / MAX_LENGTH;
    double pcc = tempSum / (sdA * sdB);
    printf("\nPearson Correlation Coefficient: %.12f\n", pcc);

    // Take ending clock time and print time taken
    end = omp_get_wtime();
    time_parallel_from_calc = (double)(end - begin_calc)*1000;
    printf("\nTime Taken (Calculation Only): %.3f ms", time_parallel_from_calc);
    time_parallel_from_init = (double)(end - begin_init)*1000;
    printf("\nTime Taken (Including Array Initialization): %.3f ms\n", time_parallel_from_init);

    // Release the memory of the arrays
    free(a);
    free(b);

    printf("\n====================== END PARALLEL ======================\n");
}
