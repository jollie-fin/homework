#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))
#define f(regle,tableau,x,y,z) (!!(regle & (1<<((tableau[x]<<2)|(tableau[y]<<1)|tableau[z]))))

int main(int argc, char **argv)
{
	int id, tot;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);

  if (argc != 4)
    MPI_Abort(MPI_COMM_WORLD,1);

  int nbetapes = atoi(argv[2]);
  int regle = atoi(argv[1]);
  int longueur = strlen(argv[3]);
  int taille = 2*nbetapes+longueur;
  
  //il est possible que la taille soit legerement trop grande si tot | taille
  int taillelocale = taille/tot + 1;
  
  //printf("%d %d %d %d %d %d %d\n", nbetapes, regle, longueur, taille, taillelocale, id, tot);
  
  //allocation. On alloue plus grand, car on inclut les cases extremes des voisins
  char *tableau1 = (char *) calloc(taillelocale+2, sizeof(char));
  char *tableau2 = (char *) calloc(taillelocale+2, sizeof(char));
  
  char *tableauancien = tableau1;
  char *tableaunouveau = tableau2;
  
  int i,ii;
  
  
  //initialisation du tableau
  for (i = 1; i <= taillelocale; i++)
  {
    int i_entree = i - 1 + id*taillelocale - nbetapes;
    if (i_entree >= 0 && i_entree < longueur)
      tableauancien[i] = argv[3][i_entree] == '1';
  }
  
  for (i = 0; i < nbetapes; i++)
  {
    MPI_Request requestg;
    MPI_Request requestd;
    MPI_Status status;
      
    if (id < tot-1)
      MPI_Isend(tableauancien+taillelocale, 1, MPI_CHAR, id+1, 0, MPI_COMM_WORLD, &requestd);
    if (id > 0)
      MPI_Isend(tableauancien+1, 1, MPI_CHAR, id-1, 0, MPI_COMM_WORLD, &requestg);
    if (id < tot-1)
      MPI_Recv(tableauancien+taillelocale+1, 1, MPI_CHAR, id+1, 0, MPI_COMM_WORLD, &status);
    if (id > 0)
      MPI_Recv(tableauancien, 1, MPI_CHAR, id-1, 0, MPI_COMM_WORLD, &status);
    
    for (ii = 1; ii <= taillelocale; ii++)
    {
      tableaunouveau[ii]=f(regle, tableauancien, ii-1, ii, ii+1);
    }
    //echange des deux tableaux
    char *tmp;
    tmp = tableauancien;
    tableauancien = tableaunouveau;
    tableaunouveau = tmp;
  }
  
  //je réduit localement d'abord, car on peut avoir dépassement d'entier avec un char lors de la réduction
  long total = 0;
  for (i = 1; i <= taillelocale; i++)
    total += tableauancien[i];
  
  //bien que la topologie du reseau soit une ligne, j'utilise MPI_Reduce par commodité, vu qu'il peut-être implémenté assez facilement (chaque processeur recoit le sous-total de son voisin de droite, ajoute sa valeur, et le transmet à gauche
  long totalreduit;
  MPI_Reduce(&total, &totalreduit, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 


  if (id == 0)
    printf("%ld\n", totalreduit);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  free(tableau1);
  free(tableau2);
  MPI_Finalize();
	return 0;
}
