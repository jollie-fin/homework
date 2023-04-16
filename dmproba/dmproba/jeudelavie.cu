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

__constant__ uint32_t w;
__constant__ uint32_t h;
__constant__ uint32_t pitch;
__constant__ uint32_t pas;
__constant__ uint8_t regle[512];

__global__ void jeu(uint8_t *nouveau, uint8_t *ancien)
{
	const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;	
	
	
	
	
	if ((x < w) && (y < h))
	{
		for (int i = 0; i < pas; i++)
		{	
			uint16_t v00 = ancien[(x-1+w)%w+pitch*((y-1+h)%h)];
			uint16_t v01 = ancien[(x-1+w)%w+pitch*((y-0+h)%h)];
			uint16_t v02 = ancien[(x-1+w)%w+pitch*((y+1+h)%h)];
			uint16_t v10 = ancien[(x-0+w)%w+pitch*((y-1+h)%h)];
			uint16_t v11 = ancien[(x-0+w)%w+pitch*((y-0+h)%h)];
			uint16_t v12 = ancien[(x-0+w)%w+pitch*((y+1+h)%h)];
			uint16_t v20 = ancien[(x+1+w)%w+pitch*((y-1+h)%h)];
			uint16_t v21 = ancien[(x+1+w)%w+pitch*((y-0+h)%h)];
			uint16_t v22 = ancien[(x+1+w)%w+pitch*((y+1+h)%h)];
	
			uint8_t resultat = regle[v00 | v01<<1 | v02<<2 | v10<<3 | v11<<4 | v12<<5 | v20<<6 | v21<<7 | v22<<8];
			nouveau[x+pitch*y] = resultat;
			syncthreads();


		}		
	/*		v00 = nouveau[(x-1+w)%w+pitch*((y-1+h)%h)];
			v01 = nouveau[(x-1+w)%w+pitch*((y-0+h)%h)];
			v02 = nouveau[(x-1+w)%w+pitch*((y+1+h)%h)];
			v10 = nouveau[(x-0+w)%w+pitch*((y-1+h)%h)];
			v11 = nouveau[(x-0+w)%w+pitch*((y-0+h)%h)];
			v12 = nouveau[(x-0+w)%w+pitch*((y+1+h)%h)];
			v20 = nouveau[(x+1+w)%w+pitch*((y-1+h)%h)];
			v21 = nouveau[(x+1+w)%w+pitch*((y-0+h)%h)];
			v22 = nouveau[(x+1+w)%w+pitch*((y+1+h)%h)];
	
			resultat = regle[v00 | v01<<1 | v02<<2 | v10<<3 | v11<<4 | v12<<5 | v20<<6 | v21<<7 | v22<<8];
	
			ancien[x+pitch*y] = resultat;
			syncthreads();
		}
	*/
	}
}


#define prend(i,num) ((i&(1<<num))>>num)

int main(int argc, char **argv)
{
	uint32_t pw = 1000;
	uint32_t ph = 1000;
	uint32_t sizex = 1024;
	uint32_t sizey = 1024;
	uint32_t ppas = 1;
	uint8_t pregle[512] = {0};


	for (int i=0; i<512; i++)
	{
		int voisinnage = 0;
		for (int j = 0; j < 9; j++)
		{
			if (j!=4)
				voisinnage += prend(i,j);
		}
		
		if (prend(i,4) && voisinnage == 2)
			pregle[i] = 1;
		else if (voisinnage == 3)
			pregle[i] = 1;
		else
			pregle[i] = 0;
	}
	
	printf("%d\n", pregle[0x49]);
	uint8_t *pnouveau;
	uint8_t *pancien;
	
	uint8_t *gnouveau;
	uint8_t *gancien;
	
	
	CUT_DEVICE_INIT(argc, argv);	
    CUDA_SAFE_CALL(cudaMalloc((void**)&gnouveau,
		      sizeof(uint8_t) * sizex * sizey));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gancien,
		      sizeof(uint8_t) * sizex * sizey));
		
	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol((const char *) "regle", pregle, 512*sizeof(uint8_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("w", &pw, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("h", &ph, sizeof(uint32_t)));    
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("pitch", &sizex, sizeof(uint32_t)));
   	CUDA_SAFE_CALL(cudaMemcpyToSymbol("pas", &ppas, sizeof(uint32_t)));    

	    
	cudaThreadSynchronize();
    cudaError_t e;
		
		      
	pnouveau = (uint8_t *) malloc(sizeof(uint8_t) * sizex * sizey);
	pancien = (uint8_t *) malloc(sizeof(uint8_t) * sizex * sizey);
	
	memset(pancien, 0, sizeof(uint8_t) * sizex * sizey);
	memset(pnouveau, 0, sizeof(uint8_t) * sizex * sizey);	
	


	pancien[0] = 1;
	pancien[1] = 1;
	pancien[2] = 1;		

	CUDA_SAFE_CALL(cudaMemcpy(gancien,
			   pancien,
			   sizeof(uint8_t) * sizex * sizey,
			   cudaMemcpyHostToDevice));
	

	CUDA_SAFE_CALL(cudaMemcpy(gnouveau,
			   pnouveau,
			   sizeof(uint8_t) * sizex * sizey,
			   cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(10, 10);
	dim3 numBlock(100, 100);	

	for (int i=0; i < 100; i++)
	{
		jeu<<< numBlock, threadsPerBlock>>>(gnouveau, gancien);	
		cudaThreadSynchronize();
		jeu<<< numBlock, threadsPerBlock>>>(gancien, gnouveau);	
		cudaThreadSynchronize();
	}

	cudaThreadSynchronize();

	e = cudaGetLastError();
	if (e != cudaSuccess) {
	printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
	exit(1);}
		
	CUDA_SAFE_CALL(cudaMemcpy(pancien,
			   gancien,
			   sizeof(uint8_t) * sizex * sizey,
			   cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaMemcpy(pnouveau,
			   gnouveau,
			   sizeof(uint8_t) * sizex * sizey,
			   cudaMemcpyDeviceToHost));

			   
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
			printf("%2x ", pancien[i*sizex+j]);	
		printf("\n");
	}
	
			printf("\n");
	
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
			printf("%2x ", pnouveau[i*sizex+j]);	
		printf("\n");
	}
	return 0;
}

