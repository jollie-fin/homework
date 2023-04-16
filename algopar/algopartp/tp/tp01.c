#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

void swap(double *a, double *b)
{
	double tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
}

void qsort(double *tab, int i, int j)
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
		tmp = tab[i]
		tab[i] = tab[position];
		tab[position] = tmp;
	}
	qsort(tab, i, position-1);
	qsort(tab, position+1, j);
}

void seqsort(double *mtx, int lmtx)
{
	qsort(mtx,0,lmtx-1);
}

void fusion(double *src1, double *src2, double *out, int lmtx)
{
	int k;
	int k1 = 0, k2 = 0;
	for (k = 0; k < 2*lmtx; k++)
	{
		if (k1 < lmtx && k2 < lmtx)
		{
			if (src[k1] < src[k2])
				out[k] = src[k1++];
			else
				out[k] = src[k2++];
		}
		else if (k1 == lmtx)
		{
			out[k] = src[k2++];
		}
		else
		{
			out[k] =src[k1++];
		}
	}
}

void sort(double *mtx, int lmtx, MPI_Comm comm)
{
	MPI_Request request;
	double *resultat = (double *) malloc(4 * lmtx * sizeof(double));
	double *tmp = resultat + 2 * lmtx;
	double *recv = resultat + 1 * lmtx;
	double *tab = resultat;

	memcpy(tab, mtx, lmtx);
	seqsort(tab, lmtx);
	
	if (id % 2 == 0 && id < tot - 1)
		MPI_Isend(tab, lmtx, MPI_DOUBLE, id + 1, 0, comm, &request);
	if (id % 2 == 1)
		MPI_Isend(tab, lmtx, MPI_DOUBLE, id - 1, 0, comm, &request);


	for (i = 1; i <= tot; i++)
	{
		if ((id + i) % 2 == 0 && id < tot-1)
		{
			MPI_Isend(tab, lmtx, MPI_DOUBLE, id + 1, 0, comm, &request);
			MPI_Recv(recv, lmtx, MPI_DOUBLE, id + 1, 0, comm);
			fusion(tab, recv, tmp, lmtx);
			tab = tmp;
			recv = tmp+lmtx;
			tmp = (tmp == resultat) ? (resultat + 2*lmtx) : resultat;
		}
		if ((id + i) % 2 == 1 && id > 0)
		{
			MPI_Isend(tab, lmtx, MPI_DOUBLE, id - 1, 0, comm, &request);
			MPI_Recv(recv, lmtx, MPI_DOUBLE, id - 1, 0, comm);
			fusion(tab, recv, tmp, lmtx);
			tab = tmp+lmtx;
			recv = tmp;
			tmp = (tmp == resultat) ? (resultat + 2*lmtx) : resultat;
		}
	}
	memcpy(mtx, tab, lmtx);
	free(resultat);
}

int main(int argc, char **argv)
{	
	int id, tot;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);


	MPI_Finalize();
	return 0;
}
