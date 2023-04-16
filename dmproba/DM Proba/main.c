#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "./dSFMT-src-2.1/dSFMT.h"
#define SMAX 100000

int simulation(double p, int X0, int Y0, int Smax);
long long nb_tests=0;
int simulation(double p, int X0, int Y0, int Smax)
{
	int X=X0,Y=Y0;
	int i = 0;
	while (i < Smax && X!=0 && Y!=0)
	{
		double de = dsfmt_gv_genrand_close_open();
		nb_tests++;
		if (X==0&&Y>0)
		{
			if (de<p/2.)
				X++;
			else if (de < p)
				Y++;
			else
				Y--;
		}
		else if (Y==0&&X>0)
		{
			if (de<p/2.)
				X++;
			else if (de < p)
				Y++;
			else
				X--;
		}
		else
		{
			if (de<1./4.)
				X++;
			else if (de < 1./2.)
				Y++;
			else if (de < 3./4.)
				X--;
			else
				Y--;
		}
		i++;
	}
	return i;

}

int main(int argc, char **argv)
{
	time_t t = time(NULL);
//	int resultat[SMAX+1] = {0};
	//#pragma omp parallel for schedule(dynamic)

	dsfmt_gv_init_gen_rand(time(NULL));
	int i;
	for (i=0; i<10000000; i++)
	{
		int s = simulation(.5, 5, 5, SMAX);
	}
/*	printf ("%f\n", s);
	int i;
	for (i = 0; i < SMAX+1; i++)
	{
		printf ("%d ", resultat[i]);
	}*/
	printf("\n");
	printf ("%f\n", (double) nb_tests/(time(NULL)-t));
	return 0;
}
