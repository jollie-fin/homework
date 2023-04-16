/**
 * @file mtgp32-cuda.cu
 *
 * @brief Sample Program for CUDA 2.2
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2009 Mutsuo Saito, Makoto Matsumoto and
 * Hiroshima University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#define __STDC_FORMAT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
extern "C" {
#include "mtgp32-fast.h"
#include "mtgp32dc-param-11213.c"
}
#define MEXP 11213
#define N MTGPDC_N
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
//#define BLOCK_NUM 32
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
    uint32_t status[N];
};

/*
 * Generator Parameters.
 */
__constant__ uint32_t param_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
__constant__ uint32_t pos_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh1_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh2_tbl[BLOCK_NUM_MAX];
/* high_mask and low_mask should be set by make_constant(), but
 * did not work.
 */

__constant__ uint32_t Smax;
__constant__ uint32_t X0;
__constant__ uint32_t Y0;
__constant__ uint32_t p;
__constant__ uint32_t p_2;
__constant__ uint32_t n_max_GPU;
__constant__ uint32_t nb_tests;

__constant__ uint32_t mask = 0xff800000;

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[LARGE_SIZE];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output, Smax, n_GPU, X0, Y0, p
 */
__device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid) {
    uint32_t X = (X1 & mask) ^ X2;
    uint32_t MAT;

    X ^= X << sh1_tbl[bid];
    Y = X ^ (Y >> sh2_tbl[bid]);
    MAT = param_tbl[bid][Y & 0x0f];
    return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ uint32_t temper(uint32_t V, uint32_t T, int bid) {
    uint32_t MAT;

    T ^= T >> 16;
    T ^= T >> 8;
    MAT = temper_tbl[bid][T & 0x0f];
    return V ^ MAT;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_read(uint32_t status[LARGE_SIZE],
			    const mtgp32_kernel_status_t *d_status,
			    int bid,
			    int tid) {
    status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
    if (tid < N - THREAD_NUM) {
	status[LARGE_SIZE - N + THREAD_NUM + tid]
	    = d_status[bid].status[THREAD_NUM + tid];
    }
    __syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void status_write(mtgp32_kernel_status_t *d_status,
			     const uint32_t status[LARGE_SIZE],
			     int bid,
			     int tid) {
    d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
    if (tid < N - THREAD_NUM) {
	d_status[bid].status[THREAD_NUM + tid]
	    = status[4 * THREAD_NUM - N + tid];
    }
    __syncthreads();
}

__device__ void deplacement(uint32_t r, uint32_t &X, uint32_t &Y)
{
	if ((X!=0) || (Y!=0))
	{
		if (X==0)
		{
			if (r < p_2)
				X++;
			else if (r < p)
				Y++;
			else
				Y--;
		}
		else if (Y==0)
		{
			if (r < p_2)
				X++;
			else if (r < p)
				Y++;
			else
				X--;
		}
		else
		{
			if (r < 4294967296. / 4.)
				X++;
			else if (r < 2.*4294967296. / 4.)
				Y++;
			else if (r < 3.*4294967296. / 4.)
				X--;
			else
				Y--;
		}
	}
}


struct Etat
{
	uint64_t tot;
	uint64_t lg;
	uint64_t nb_chemin;
	uint64_t prob;
	uint64_t test;
	uint32_t X;
	uint32_t Y;
	uint32_t i_chemin;
};

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
				     Etat* d_etat) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

    uint64_t tot = d_etat[blockDim.x * bid + tid].tot;
    uint64_t lg = d_etat[blockDim.x * bid + tid].lg;
    uint64_t nb = d_etat[blockDim.x * bid + tid].nb_chemin;
    uint64_t test = d_etat[blockDim.x * bid + tid].test;
    uint64_t prob = d_etat[blockDim.x * bid + tid].prob;
	uint32_t X = d_etat[blockDim.x * bid + tid].X;
	uint32_t Y = d_etat[blockDim.x * bid + tid].Y;
	uint32_t i_chemin = d_etat[blockDim.x * bid + tid].i_chemin;
	uint32_t j;

	j=0;
	while ((test != nb_tests) && (j < n_max_GPU))
	{
		j++;

		if ((i_chemin<Smax) && ((X!=0)||(Y!=0)))
    	{
			r = para_rec(status[LARGE_SIZE - N + tid],
						 status[LARGE_SIZE - N + tid + 1],
						 status[LARGE_SIZE - N + tid + pos],
						 bid);
			status[tid] = r;
			o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
			deplacement(o,X,Y);
			i_chemin++;
		}
		if ((i_chemin<Smax) && ((X!=0)||(Y!=0)))
		{
			r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
					 status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
					 status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
					 bid);
			status[tid + THREAD_NUM] = r;
			o = temper(r,
				   status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				   bid);
   			deplacement(o,X,Y);
   			i_chemin++;
		}
		if ((i_chemin<Smax) && ((X!=0)||(Y!=0)))
		{		
			r = para_rec(status[2 * THREAD_NUM - N + tid],
					 status[2 * THREAD_NUM - N + tid + 1],
					 status[2 * THREAD_NUM - N + tid + pos],
					 bid);
			status[tid + 2 * THREAD_NUM] = r;
			o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
			deplacement(o,X,Y);

			i_chemin ++;
		}
		else
		{
			prob+=i_chemin<Smax;
		    test++;	
		    tot+=i_chemin;
			if (i_chemin < Smax)			nb++;
			if (i_chemin < Smax)			lg+=i_chemin;
			i_chemin=0;
			X=X0;
			Y=Y0;

		}
    }


	d_etat[blockDim.x * bid + tid].nb_chemin = nb;	
	__syncthreads();
	d_etat[blockDim.x * bid + tid].tot = tot;	
	__syncthreads();
	d_etat[blockDim.x * bid + tid].lg = lg;
	__syncthreads();	
	d_etat[blockDim.x * bid + tid].test = test;
	__syncthreads();
	d_etat[blockDim.x * bid + tid].i_chemin = i_chemin;
	__syncthreads();
	d_etat[blockDim.x * bid + tid].prob = prob;
	__syncthreads();
	d_etat[blockDim.x * bid + tid].X = X;		
	__syncthreads();
	d_etat[blockDim.x * bid + tid].Y = Y;		
	__syncthreads();

	
    status_write(d_status, status, bid, tid);
}


/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP32 parameters.
 */

void make_constant2(uint32_t h_Smax, uint32_t h_n_max_GPU, uint32_t h_X0, uint32_t h_Y0, double dp, uint32_t h_nb_tests)
{
    uint32_t h_p = 4294967296.*dp;
    uint32_t h_p_2 = 4294967296.*dp/2.;

    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("n_max_GPU", &h_n_max_GPU, sizeof(h_n_max_GPU)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("X0", &h_X0, sizeof(h_X0)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("Y0", &h_Y0, sizeof(h_Y0)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("p", &h_p, sizeof(h_p)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("p_2", &h_p_2, sizeof(h_p_2)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("Smax", &h_Smax, sizeof(h_Smax)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("nb_tests", &h_nb_tests, sizeof(h_nb_tests)));    

}
	
void make_constant(const mtgp32_params_fast_t params[], int block_num) {

    const int size1 = sizeof(uint32_t) * block_num;
    const int size2 = sizeof(uint32_t) * block_num * TBL_SIZE;

    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_param_tbl;
    uint32_t *h_temper_tbl;
    uint32_t *h_single_temper_tbl;
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_single_temper_tbl = (uint32_t *)malloc(size2);
   if (h_pos_tbl == NULL
	|| h_sh1_tbl == NULL
	|| h_sh2_tbl == NULL
	|| h_param_tbl == NULL
	|| h_temper_tbl == NULL
	|| h_single_temper_tbl == NULL
	) {
	printf("failure in allocating host memory for constant table.\n");
	exit(1);
    }
    for (int i = 0; i < block_num; i++) {
	h_pos_tbl[i] = params[i].pos;
	h_sh1_tbl[i] = params[i].sh1;
	h_sh2_tbl[i] = params[i].sh2;
	for (int j = 0; j < TBL_SIZE; j++) {
	    h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j];
	    h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j];
	    h_single_temper_tbl[i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
	}
    }
    // copy from malloc area only

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(param_tbl, h_param_tbl, size2));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(temper_tbl, h_temper_tbl, size2));
       
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_single_temper_tbl);
}

#include "mtgp-cuda-common.c"
#include "mtgp32-cuda-common.c"

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */

 FILE *fichier;
 
void calcul(int block_num, mtgp32_kernel_status_t *d_status, uint64_t Smax, uint32_t n_max_GPU, uint32_t nb_tests, uint32_t X0, uint32_t Y0, double p)
{
    Etat* d_etat;
	Etat* h_etat;
    unsigned int timer = 0;
    uint64_t tot = 0ull;
    uint64_t nb = 0ull;
    uint64_t lg = 0ull;    
    uint64_t test = 0ull;
    uint64_t prob = 0ull;
    
    cudaError_t e;
    float gputime;
	make_constant2(Smax, n_max_GPU, X0, Y0, p, nb_tests);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_etat, sizeof(Etat) * block_num*THREAD_NUM));
   
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    h_etat = (Etat *) malloc(sizeof(Etat) * THREAD_NUM*block_num);
/*    if (h_data == NULL) {
	printf("failure in allocating host memory for output data.\n");
	exit(1);
    }*/
    CUT_SAFE_CALL(cutStartTimer(timer));
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

	for (int i = 0; i < THREAD_NUM*block_num; i++)
	{
		h_etat[i].i_chemin=0;
		h_etat[i].X=X0;
		h_etat[i].Y=Y0;						
		h_etat[i].tot=0;
		h_etat[i].lg=0;
		h_etat[i].nb_chemin=0;		
		h_etat[i].test=0;																						
	}

	CUDA_SAFE_CALL(
	cudaMemcpy(d_etat,
		   h_etat,
		   sizeof(Etat) * THREAD_NUM*block_num,
		   cudaMemcpyHostToDevice));
	cudaThreadSynchronize();		   

	bool fin = true;

    printf("\nSmax=%lld, X0=%d, Y0=%d, p=%f\n",Smax,X0,Y0,p);
	do
	{
		mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
		d_status, d_etat);
		cudaThreadSynchronize();

		e = cudaGetLastError();
		if (e != cudaSuccess) {
		printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
		exit(1);
		}

		CUDA_SAFE_CALL(
		cudaMemcpy(h_etat,
			   d_etat,
			   sizeof(Etat) * THREAD_NUM*block_num,
			   cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();			   
		fin = true;

		lg=0;
		nb=0;
		tot=0;
		test=0;
		prob=0;
		for (int i = 0; i < THREAD_NUM*block_num; i++)
		{
			tot+=h_etat[i].tot;
			test+=h_etat[i].test;
			prob+=h_etat[i].prob;
			fin = fin && (h_etat[i].test == nb_tests);
			lg+=h_etat[i].lg;
			nb+=h_etat[i].nb_chemin;						
		}	

    gputime = cutGetTimerValue(timer);
    printf("\rtot %E, vtot %E, test %E, vtest %15f lmoy %f p' %f %2d%% temps %f", double(tot), double(tot) / (gputime * 0.001), double(test), double(test) / (gputime * 0.001), double(lg)/double(nb), double(prob)/double(test), int(100.*double(test)/double(THREAD_NUM*block_num*nb_tests)),gputime*0.001);
		fflush(stdout);	    
		usleep(100000);
	}while (!fin);


	fprintf(fichier,"X0 %d; Y0 %d; Smax %lld; p %f; temps moyen %f p' %f\n",X0,Y0,Smax,p,double(lg)/double(nb),double(prob)/double(test));
	fflush(fichier);

    CUT_SAFE_CALL(cutDeleteTimer(timer));
    free(h_etat);
    CUDA_SAFE_CALL(cudaFree(d_etat));    
}

int main(int argc, char** argv)
{
	fichier = fopen("exercice1.txt","ar");

#define MODE 2
#ifdef MODE
#if MODE == 0 //calcul pour k
	uint64_t S[5]={10,100,1000,10000,100000};
	uint32_t n[5]={10,10,10,10,10};
	uint32_t x[50]={};
	uint32_t y[50]={};
	double p[1] = {0.9};
	for (int i=0; i<50; i++)
	{
		x[i] = i+1;
		y[i] = 0;
	}
#elif MODE==1 //cas general
	uint64_t S[7]={10,100,1000,10000,100000,1000000,10000000};
	uint32_t n[7]={10,10,10,10,10,5,5};
	uint32_t x[25]={};
	uint32_t y[25]={};
	double p[1] = {0.9};
	uint32_t data[5]={0,1,2,5,10};
	for (int i=0; i<5; i++)
	for (int j=0; j<5; j++)
	{
		x[5*i+j] = data[i];
		y[5*i+j] = data[j];
	}
#elif MODE==2 //test
	uint64_t S[1]={10000};
	uint32_t n[1]={100};
	uint32_t x[1]={1};
	uint32_t y[1]={1};
	double p[1] = {0.9};
#else
#error "choisir MODE"
#endif
#else
#error "choisir MODE"
#endif

	CUT_DEVICE_INIT(argc, argv);
   int block_num = get_suitable_block_num(sizeof(uint32_t),
				   THREAD_NUM,
				   LARGE_SIZE);
    mtgp32_kernel_status_t *d_status;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * block_num));
    make_constant(MTGPDC_PARAM_TABLE, block_num);
    make_kernel_data(d_status, MTGPDC_PARAM_TABLE, block_num);

	for (uint32_t i=0; i<sizeof(x)/sizeof(*x);i++)
		for (uint32_t ii=0; ii<sizeof(p)/sizeof(*p);ii++)
			for (uint32_t iii=0; iii<sizeof(S)/sizeof(*S);iii++)
					calcul(block_num,d_status,S[iii],50000,n[iii],x[i],y[i],p[ii]);
		
    //finalize
    CUDA_SAFE_CALL(cudaFree(d_status));
#ifdef NEED_PROMPT
    CUT_EXIT(argc, argv);
#endif
	fclose(fichier);
}
