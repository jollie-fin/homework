#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void swap(double *a, double *b)
{
	double tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
}

void qsortmaison(double *tab, int i, int j)
{
	if (i >= j) return;

	double pivot = tab[i];
	int position = i;
	int k;
	for (k = i +1; k<= j; k++)
	{
		if (tab[k] < pivot)
		{
			double tmp;
			position++;
			tmp = tab[k];
			tab[k] = tab[position];
			tab[position] = tmp;
		}
	}
	{
		double tmp;
		tmp = tab[i];
		tab[i] = tab[position];
		tab[position] = tmp;
	}
	qsortmaison(tab, i, position-1);
	qsortmaison(tab, position+1, j);
}

void seqsortmaison(double *mtx, int lmtx)
{
	qsortmaison(mtx,0,lmtx-1);
}

void fusion(double *src1, double *src2, double *out, int lmtx)
{
	int k;
	int k1 = 0, k2 = 0;
	for (k = 0; k < 2*lmtx; k++)
	{
		if (k1 < lmtx && k2 < lmtx)
		{
			if (src1[k1] < src2[k2])
				out[k] = src1[k1++];
			else
				out[k] = src2[k2++];
		}
		else if (k1 == lmtx)
		{
			out[k] = src2[k2++];
		}
		else
		{
			out[k] =src1[k1++];
		}
	}
}

void sort(double *mtx, int lmtx, MPI_Comm comm)
{
	int id, tot;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);

	MPI_Request request;
	MPI_Status status;
	double *resultat = (double *) malloc(4 * lmtx * sizeof(double));
	double *tmp = resultat + 2 * lmtx;
	double *recv = resultat + 1 * lmtx;
	double *tab = resultat;

	memcpy(tab, mtx, lmtx*sizeof(double));
	seqsortmaison(tab, lmtx);
	
	int i;
	for (i = 0; i < tot-1; i++)
	{
		if ((id + i) % 2 == 0 && id < tot-1)
		{
			MPI_Isend(tab, lmtx, MPI_DOUBLE, id + 1, 0, comm, &request);
			MPI_Recv(recv, lmtx, MPI_DOUBLE, id + 1, 0, comm, &status);
			fusion(tab, recv, tmp, lmtx);
			tab = tmp;
			recv = tmp+lmtx;
			tmp = (tmp == resultat) ? (resultat + 2*lmtx) : resultat;
		}
		if ((id + i) % 2 == 1 && id > 0)
		{
			MPI_Isend(tab, lmtx, MPI_DOUBLE, id - 1, 0, comm, &request);
			MPI_Recv(recv, lmtx, MPI_DOUBLE, id - 1, 0, comm, &status);
			fusion(tab, recv, tmp, lmtx);
			tab = tmp+lmtx;
			recv = tmp;
			tmp = (tmp == resultat) ? (resultat + 2*lmtx) : resultat;
		}
	}
	memcpy(mtx, tab, lmtx*sizeof(double));
	free(resultat);
}

int main(int argc, char **argv)
{
	int id, tot;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);

	srand(getpid()+time(NULL));
	double tab[256];
	int i;	
	for (i =0; i < 256; i++)
		tab[i] = ((double) rand())/((double) RAND_MAX);
	sort(tab, 256, MPI_COMM_WORLD);

	double *resultat = NULL;
	if (id == 0)
		resultat = (double *) malloc(256*tot*sizeof(double));

	MPI_Gather(tab, 256, MPI_DOUBLE, resultat, 256, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (id == 0)
	{
		for (i =0; i < tot*256; i++)
		{
			printf("%lf ",resultat[i]);
			if (i % 10 == 9) printf("\n");
		}
		printf("\n");
		free (resultat);
	}
	MPI_Finalize();
	return 0;
}
