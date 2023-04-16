/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements Mersenne Twister random number generator 
 * and Cartesian Box-Muller transformation on the GPU.
 * See supplied whitepaper for more explanations.
 */


// Utilities and system includes
#include <shrUtils.h>
#include <cutil_inline.h>

#include "MersenneTwister.h"

///////////////////////////////////////////////////////////////////////////////
// Common host and device function 
///////////////////////////////////////////////////////////////////////////////
//ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//floor(a / b)
extern "C" int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b){
    return a - a % b;
}

///////////////////////////////////////////////////////////////////////////////
// Reference MT front-end and Box-Muller transform
///////////////////////////////////////////////////////////////////////////////
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Random, int NPerRng, unsigned int seed);
extern "C" void BoxMullerRef(float *h_Random, int NPerRng);

///////////////////////////////////////////////////////////////////////////////
// Fast GPU random number generator and Box-Muller transform
///////////////////////////////////////////////////////////////////////////////
#include "MersenneTwister_kernel.cu"

///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
const int    PATH_N = 24000000;
const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;

const unsigned int SEED = 777;

//#define DO_BOXMULLER
///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int N = 1000000000;
	int X0 = 5;
	int Y0 = 5;
	int Smax = 1000000;
	int p = (double) 4294967296. * .1;
	int p_2 = (double) 4294967296. * .1 / 2.;
	
    // Start logs
    shrSetLogFileName ("MersenneTwister.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    float
        *d_Rand;

    float
        *h_RandCPU,
        *h_RandGPU;

    double
        rCPU, rGPU, delta, sum_delta, max_delta, sum_ref, L1norm, gpuTime;

    int i, j;
    unsigned int hTimer;


    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError( cutCreateTimer(&hTimer) );

    shrLog("Initializing data for %i samples...\n", PATH_N);
        h_RandGPU  = (float *)malloc((Smax+1) * sizeof(int));
        cutilSafeCall( cudaMalloc((void **)&d_Rand, (Smax+1) * sizeof(int)) );

    shrLog("Loading CPU and GPU twisters configurations...\n");
        const char *raw_path = shrFindFilePath("MersenneTwister.raw", argv[0]);
        const char *dat_path = shrFindFilePath("MersenneTwister.dat", argv[0]);
        initMTRef(raw_path);
        loadMTGPU(dat_path);
        seedMTGPU(SEED);

    shrLog("Generating random numbers on GPU...\n\n");
	int numIterations = 10;

	cutilSafeCall( cudaThreadSynchronize() );
	cutilCheckError( cutResetTimer(hTimer) );
	cutilCheckError( cutStartTimer(hTimer) );

    RandomGPU<<<32, 128>>>(d_Rand, N, Smax, p, p_2, X0, Y0);
    cutilCheckMsg("RandomGPU() execution failed\n");

    cutilSafeCall( cudaThreadSynchronize() );
    cutilCheckError( cutStopTimer(hTimer) );

    shrLog("\nReading back the results...\n");
    cutilSafeCall( cudaMemcpy(h_RandGPU, d_Rand, Smax * sizeof(int), cudaMemcpyDeviceToHost) );

	long long total = 0;
	for (i = 0; i <= Smax; i++)
	{
		total += h_RandGPU[i];
	}
	printf("nb_tests=%lld, p'=%f\n", total, (double) ((long long) total - h_RandGPU[Smax]) / total);

        cutilSafeCall( cudaFree(d_Rand) );
        free(h_RandGPU);
        free(h_RandCPU);

    cutilCheckError( cutDeleteTimer( hTimer) );

    cudaThreadExit();

    shrEXIT(argc, (const char**)argv);
}
