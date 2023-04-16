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
__constant__ uint32_t mask = 0xff800000;

/*variables personnelles*/
__constant__ uint32_t n_GPU;
__constant__ uint32_t matrice[20];
__constant__ uint32_t distribution_initiale[4];
__constant__ uint32_t NChem;


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

/*change la position du chat en fonction de la valeur aleatoire r, et de la position precedente*/
__device__ void deplacement(uint32_t r, uint32_t &position)
{
	if (r < matrice[position+0])
	{
		position = 0;
	}
	else if (r < matrice[position+5])
	{
		position = 1;
	}
	else if (r < matrice[position+10])
	{
		position = 2;
	}
	else if (r < matrice[position+15])
	{
		position = 3;
	}
	else
	{
		position = 4;
	}
}

/*initialise la position du chat en fonction de la valeur aleatoire r*/
__device__ void initialise(uint32_t r, uint32_t &position)
{
	if (r < distribution_initiale[0])
	{
		position = 0;
	}
	else if (r < distribution_initiale[1])
	{
		position = 1;
	}
	else if (r < distribution_initiale[2])
	{
		position = 2;
	}
	else if (r < distribution_initiale[3])
	{
		position = 3;
	}
	else
	{
		position = 4;
	}
}

struct Etat
{
	uint64_t tot;
	uint64_t test;
	uint64_t frequence[5];
	uint32_t fin[5];
};

/*
	calcul proprement dit
 */
__global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
				     Etat* d_etat) {
	/*recupere les identifiants du thread*/
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int pos = pos_tbl[bid];
    uint32_t r;
    uint32_t o;

    // copy status data from global memory to shared memory.
    status_read(status, d_status, bid, tid);

	/*variables utiles recuperer du tableau d'etat*/
	/*le thread incrementera ces valeurs en fonction des calculs qu'il effectuera*/
    uint64_t tot = d_etat[blockDim.x * bid + tid].tot; //total du nombre de deplacement effectues par le thread lors de cette simulation
    uint64_t test = d_etat[blockDim.x * bid + tid].test;//total du nombre de marches effectuees par le thread lors de cette simulation
    uint32_t frequence[3][5] = {0};//frequence de passage sur les divers plats
    uint32_t fin[3][5] = {0};//frequence de arret sur les divers plats
	uint32_t position[3] = {0};//position actuelle du chat

	uint32_t j;
	uint32_t i;
	
	for (i=0;i<5;i++)
	{
		frequence[1][i] = 0;
		fin[1][i] = 0;
		frequence[2][i] = 0;
		fin[2][i] = 0;

		frequence[0][i] = d_etat[blockDim.x * bid + tid].frequence[i];
		__syncthreads();
		fin[0][i] = d_etat[blockDim.x * bid + tid].fin[i];
		__syncthreads();
	}


	/*calcul proprement dit de trois marches simultanee, pour diminuer les dependances de variables*/
	for (j = 0; j < n_GPU; j++)
	{
		/*initialise position[0]*/
		/*calcul d'une valeur pseudo aleatoire dans o*/
		r = para_rec(status[LARGE_SIZE - N + tid],
					 status[LARGE_SIZE - N + tid + 1],
					 status[LARGE_SIZE - N + tid + pos],
					 bid);
		status[tid] = r;
		o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
		/*sans commentaire*/
		initialise(o, position[0]);
		
		/*initialise position[1]*/
		r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
				 status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
				 status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
				 bid);
		status[tid + THREAD_NUM] = r;
		o = temper(r,
			   status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
			   bid);
		initialise(o, position[1]);

		/*initialise position[2]*/		
		r = para_rec(status[2 * THREAD_NUM - N + tid],
				 status[2 * THREAD_NUM - N + tid + 1],
				 status[2 * THREAD_NUM - N + tid + pos],
				 bid);
		status[tid + 2 * THREAD_NUM] = r;
		o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
		initialise(o, position[2]);

		/*calcul les marches*/
		for (i = 0; i < NChem; i++)
		{
			/*deplace le chat 0*/
			r = para_rec(status[LARGE_SIZE - N + tid],
						 status[LARGE_SIZE - N + tid + 1],
						 status[LARGE_SIZE - N + tid + pos],
							 bid);
			status[tid] = r;
			o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);
			deplacement(o, position[0]);
			frequence[0][position[0]]++;
			
			/*deplace le chat 1*/
			r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
					 status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
					 status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
					 bid);
			status[tid + THREAD_NUM] = r;
			o = temper(r,
				   status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				   bid);
   			deplacement(o,position[1]);
   			frequence[1][position[1]]++;

			/*deplace le chat 2*/			
			r = para_rec(status[2 * THREAD_NUM - N + tid],
					 status[2 * THREAD_NUM - N + tid + 1],
					 status[2 * THREAD_NUM - N + tid + pos],
					 bid);
			status[tid + 2 * THREAD_NUM] = r;
			o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
   			deplacement(o,position[2]);
   			frequence[2][position[2]]++;
		}
		/*conclut*/
		fin[0][position[0]]++;
		fin[1][position[1]]++;
		fin[2][position[2]]++;				
		test+=3;
		tot+=3*NChem;
    }

	/*retourne les valeurs*/
	d_etat[blockDim.x * bid + tid].test = test;	
	__syncthreads();
	d_etat[blockDim.x * bid + tid].tot = tot;	
	__syncthreads();
	for (i=0;i<5;i++)
	{
		d_etat[blockDim.x * bid + tid].frequence[i] = frequence[0][i] + frequence[1][i] + frequence[2][i];
		__syncthreads();
		d_etat[blockDim.x * bid + tid].fin[i] = fin[0][i] + fin[1][i] + fin[2][i];
		__syncthreads();
	}

	
    status_write(d_status, status, bid, tid);
}

/*copie les donnees constantes necessaires a la simulation*/
/*NChem correspond a N
  n_GPU correspond au nombre de marches par thread
  matrice correspond au tableau de seuil de la matrice
  distribution initiale correspond au tableau de seuil pour l'initialisation de premiere position*/
void make_constant2(uint32_t h_NChem, uint32_t h_n_GPU, uint32_t h_matrice[16], uint32_t h_distribution_initiale[4])
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("n_GPU", &h_n_GPU, sizeof(h_n_GPU)));
//    CUDA_SAFE_CALL(cudaMemcpyToSymbol("nb_tests", &h_nb_tests, sizeof(h_nb_tests)));    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("matrice", h_matrice, 20*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("distribution_initiale", h_distribution_initiale, 4*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("NChem", &h_NChem, sizeof(h_NChem)));    

}


/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP32 parameters.
 */


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

/*reliquat de MTGP*/
#include "mtgp-cuda-common.c"
#include "mtgp32-cuda-common.c"

FILE *fichier;

/*transforme la matrice de probabilite en tableau de seuil*/
void transforme_matrice(uint32_t matrice_transformee[20], double matrice[5][5])
{
	for (int i = 0; i < 5; i++)
	{
		matrice_transformee[i] = matrice[i][0] * 4294967296.;
		
		for (int j = 1; j < 4; j++)
			matrice_transformee[i+5*j] = uint32_t(matrice[i][j] * 4294967296.) + matrice_transformee[i+5*(j-1)];
	}
}

void transforme_distrib(uint32_t distrib_transformee[4],double distribution_initiale[5])
{
	distrib_transformee[0] = uint32_t(distribution_initiale[0] * 4294967296.);
	for (int i = 1; i < 4; i++)
		distrib_transformee[i] = uint32_t(distribution_initiale[i] * 4294967296.) + distrib_transformee[i-1];
}



/*initialise, lance et traite le calcul*/
/*block_num correspond au nombre de blocs de threads utilisés
  d_status sert pour mersenne twister
  NChem correspond a N
  n_max_GPU correspond au nombre de marches par thread par lancé de calcul
  nb_tests correspond au nombre de lancés de calcul
  matrice correspond à la matrice de la chaine de markov
  distribution initiale correspond a la distribution de probabilite pour l'initialisation de premiere position*/ 
void calcul(int block_num, mtgp32_kernel_status_t *d_status, uint32_t Nchem, uint32_t n_max_GPU, uint32_t nb_tests, double matrice[5][5], double distribution_initiale[5])
{
    Etat* d_etat;
	Etat* h_etat;
    unsigned int timer = 0;
    uint64_t tot = 0ull; //nombre de deplacements
    uint64_t test = 0ull; //nombre de tests
	uint32_t fin[5] = {0}; //Ffin * test
	uint64_t frequence[5] = {0}; //FGlobal * tot

/*transforme la matrice de probabilite (de type double) en un tableau de seuils (entiers)*/
	uint32_t matrice_transformee[20];
	transforme_matrice(matrice_transformee,matrice);
/*transforme la distribution de probabilite (de type double) en un tableau de seuils (entiers)*/	
	uint32_t distrib_transformee[4];	
	transforme_distrib(distrib_transformee,distribution_initiale);	

	
    cudaError_t e;
    float gputime;
/*transmet les valeurs constantes*/
	make_constant2(Nchem, n_max_GPU, matrice_transformee, distrib_transformee);


/*cree le timer*/   
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    CUT_SAFE_CALL(cutStartTimer(timer));
    if (cudaGetLastError() != cudaSuccess) {
	printf("error has been occured before kernel call.\n");
	exit(1);
    }

/*alloue, initialise et transmet le tableau qui recuperera les donnees des threads*/
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_etat, sizeof(Etat) * block_num*THREAD_NUM));
    h_etat = (Etat *) malloc(sizeof(Etat) * THREAD_NUM*block_num);

	for (int i = 0; i < THREAD_NUM*block_num; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			h_etat[i].frequence[j]=0;
			h_etat[i].fin[j]=0;				
		}
		h_etat[i].tot=0;
		h_etat[i].test=0;																						
	}

	CUDA_SAFE_CALL(
	cudaMemcpy(d_etat,
		   h_etat,
		   sizeof(Etat) * THREAD_NUM*block_num,
		   cudaMemcpyHostToDevice));
	cudaThreadSynchronize();		   


/*affiche la distribution initiale*/
	printf("\nN=%d Nexp=%lld\n",Nchem,(uint64_t) n_max_GPU*nb_tests*3*block_num*THREAD_NUM);
	fprintf(fichier,"\nN=%d Nexp=%lld\n",Nchem,(uint64_t) n_max_GPU*nb_tests*3*block_num*THREAD_NUM);
	printf("Distribution\n");
	fprintf(fichier,"Distribution initiale\n");
	for (int i = 0; i < 5; i++)
	{
		printf("%7.4f ",distribution_initiale[i]);
		fprintf(fichier,"%7.4f ",distribution_initiale[i]);		
	}
	printf("\n");
	fprintf(fichier,"\n");

/*lance le calcul*/
	for (uint32_t k = 0; k < nb_tests; k++)
	{
	/*lance les threads*/
		mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
		d_status, d_etat);
		cudaThreadSynchronize();

		e = cudaGetLastError();
		if (e != cudaSuccess) {
		printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
		exit(1);
		}

	/*recupere les resultats temporaires*/
		CUDA_SAFE_CALL(
		cudaMemcpy(h_etat,
			   d_etat,
			   sizeof(Etat) * THREAD_NUM*block_num,
			   cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();			   

		tot=0;
		test=0;

		for (int j = 0; j < 5; j++)
		{
			frequence[j] = 0;
			fin[j] = 0;
		}
		for (int i = 0; i < THREAD_NUM*block_num; i++)
		{
			tot+=h_etat[i].tot;
			test+=h_etat[i].test;
			for (int j = 0; j < 5; j++)
			{
				frequence[j] += h_etat[i].frequence[j];
				fin[j] += h_etat[i].fin[j];
			}			
		}

		gputime = cutGetTimerValue(timer);

		/*affiche ces resultats*/
		printf("\rtot %E , vtot %E, test %E, vtest %15f FGlobal=(", double(tot), double(tot) / (gputime * 0.001), double(test), double(test) / (gputime * 0.001));
		for (int j = 0; j < 5; j++)
		{
			printf("%7.4f ", double(frequence[j])/double(tot));
		}
		printf(") Ffin(");
		for (int j = 0; j < 5; j++)
		{
			printf("%7.4f ", double(fin[j])/double(test));
		}
		printf(" ) %2d%% temps %f;", int(100.*double(k)/double(nb_tests)),gputime*0.001);    
		fflush(stdout);	    
		usleep(100000);
	}

	fprintf(fichier,"Fglob=(");
	for (int j = 0; j < 5; j++)
	{
		fprintf(fichier,"%7.4f ", double(frequence[j])/double(tot));
	}
	fprintf(fichier,") Ffin=(");
	for (int j = 0; j < 5; j++)
	{
		fprintf(fichier,"%7.4f ", double(fin[j])/double(test));
	}
	fprintf(fichier,")\n");    


	fflush(fichier);



    CUT_SAFE_CALL(cutDeleteTimer(timer));
    free(h_etat);
    CUDA_SAFE_CALL(cudaFree(d_etat));    
}

int main(int argc, char** argv)
{
	fichier = fopen("exercice2.txt","ar");
	//Nchem correspond a N dans l'exercice
	//Nexp*3 correspond au nombre de marches lance par un thread, soit pour une GTX275 3*300000*Nexp marches lancees
	//distrib et matrice correspondent à la distribution de probabilitées initiale, et a la matrice de la chaine. Il faut que la somme des quatre premières valeurs d'une ligne fasse strictement moins de 1. pour des raisons de dépassement d'entier
#define MODE 0
#ifdef MODE
#if MODE == 0 //cas general
	double distrib[10][5] = {{.99,.0,.0,.0,.01},
							 {0.,.99,.0,.0,.01},
							 {0.,0.,.99,.0,.01},
							 {0.,0.,0.,.99,.01},
							 {0.,0.,0.,.01,.99},							 
						  	 {.5  ,.5  ,0.  ,0.  ,0.  },
							{.9  ,.025,.025,.025,.025},
							{.2  ,.2  ,.2  ,.2  ,.2  },
							{.0  ,.0  ,.33 ,.33 ,.33 },
							{.0  ,.01 ,.0  ,.0  ,.99 }};


	double matrice[5][5] = {{0.1,0.4,0.2,0.05,0.25},
	                        {0.6,0.2,0.05,0.05,0.1},
	                        {0.1,0.2,0.2,0.3,0.2},
	                        {0.05,0.2,0.4,0.25,0.1},
	                        {0.2,0.2,0.2,0.05,0.35}};
	int Nchem[3]={100,1000,10000};
	int Nexp[3]={50000,5000,500};
#elif MODE == 1 //connexite
	double distrib[2][5] = {{.999,0.,0.,0.,0.},
							{0.,0.,.999,0.,0.}};


	double matrice[5][5] = {{.499999,.499999,.0,.0,.0},
							{.499999,.499999,.0,.0,.0},
							{.0,.0,.499999,.499999,.0},
							{.0,.0,.499999,.499999,.0},
							{.0,.0,.0,.0,1.}};							
	int Nchem[1]={10000};
	int Nexp[1]={5};
#elif MODE == 2 //periodicite
	double distrib[2][5] = {{.999,.0,.0,.0,.001},
							 {0.,.999,.0,.0,.01}};


	double matrice[5][5] = {{0.,.4999,0.,.4999,0.},
	                        {.4999,0.,.4999,0.,0.},
	                        {0.,.4999,0.,.4999,0.},
	                        {.4999,0.,.4999,0.,0.},
	                        {0.,0.,0.,0.,1.}};
	int Nchem[2]={100,101};
	int Nexp[2]={5,5};

#else
#error "choisir MODE"
#endif
#else
#error "choisir MODE"
#endif
/*initialisation de CUDA*/
	CUT_DEVICE_INIT(argc, argv);
   int block_num = get_suitable_block_num(sizeof(uint32_t),
				   THREAD_NUM,
				   LARGE_SIZE);
    mtgp32_kernel_status_t *d_status;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * block_num));
    make_constant(MTGPDC_PARAM_TABLE, block_num);
    make_kernel_data(d_status, MTGPDC_PARAM_TABLE, block_num);

/*affichage des donnees*/
	fprintf(fichier,"\n\nMatrice\n");
	printf("\n\nMatrice\n");
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			fprintf(fichier,"%7.4f ",matrice[i][j]);		
			printf("%7.4f ",matrice[i][j]);
		}
		fprintf(fichier,"\n");
		printf("\n");
	}

/*calcul*/
	for (uint32_t i = 0; i < sizeof(distrib)/sizeof(*distrib); i++)
	{
		for (uint32_t ii=0; ii<sizeof(Nchem)/sizeof(*Nchem); ii++)
				calcul(block_num,d_status,Nchem[ii],Nexp[ii],5,matrice,distrib[i]);
	}	
/*ferme cuda*/
    CUDA_SAFE_CALL(cudaFree(d_status));
	fclose(fichier);
}
