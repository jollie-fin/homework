#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))

int main(int argc, char **argv)
{
	int id, tot;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);

  if (argc != 4)
    MPI_Abort(MPI_COMM_WORLD,1);

  int taille = atoi(argv[1]);
  int nbetapes = atoi(argv[2]);
  double densite = atof(argv[3]);
  int largeur_grille = 1;
  while ((largeur_grille+1) * (largeur_grille+1) <= tot)
    largeur_grille++;
  largeur_grille = MIN(largeur_grille, taille);

  int actif  = id < largeur_grille*largeur_grille;
  
  if (actif)
  {
    int taillelocale = (taille+largeur_grille-1)/largeur_grille;
    int x_grille = id%largeur_grille;
    int y_grille = id/largeur_grille;
    
    int gauche = (x_grille != 0)                           ? id-1              : id+largeur_grille-1;
    int droite = (x_grille != largeur_grille-1) ? id+1              : id-largeur_grille+1;
    int haut   = (y_grille != 0)                           ? id - largeur_grille : id + largeur_grille*(largeur_grille-1);
    int bas    = (y_grille != largeur_grille-1) ? id + largeur_grille : id - largeur_grille*(largeur_grille-1);

    int largeur = MIN(taillelocale, taille-(id%largeur_grille)*taillelocale);
    int hauteur = MIN(taillelocale, taille-(id/largeur_grille)*taillelocale);

    //printf ("%d: actif, grille de taille %dx%d\n", id, largeur, hauteur);
    
    int offset = largeur+2;
    
    //allocation. On alloue plus grand, car on inclut les cases extremes des voisins
    char *tableau = (char *) calloc(2*offset*(hauteur+2), sizeof(char));
    char *tableauancien = tableau;
    char *tableaunouveau = tableau + offset*(hauteur+2);
    char *temporaire = (char *) calloc(4*hauteur+8, sizeof(char));
    
    char *temporairegaucheenvoie = temporaire;
    char *temporairedroiteenvoie = temporaire+(hauteur+2);
    char *temporairegaucherecoit = temporaire+(hauteur+2) * 2;
    char *temporairedroiterecoit = temporaire+(hauteur+2) * 3;
    
    int i,ii,iii;
    
    srand(time(NULL)+id);
    
    for (ii = 1*offset; ii <= hauteur*offset; ii+=offset)
      for (iii = ii+1; iii <= ii+largeur; iii++)
        tableauancien[iii] = (((double) rand()) / ((double) RAND_MAX)) < densite;
    
    MPI_Request request;
    MPI_Status status;
      
    
    for (i = 0; i < nbetapes; i++)
    {
      if (y_grille > 0)
      {
        MPI_Isend(tableauancien+offset+1, largeur, MPI_CHAR, haut, 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        MPI_Recv(tableaunouveau+1, largeur, MPI_CHAR, haut, 0, MPI_COMM_WORLD, &status);
      }
      
      if (y_grille < largeur_grille - 1)
      {
        MPI_Recv(tableauancien+offset*(hauteur+1)+1, largeur, MPI_CHAR, bas, 0, MPI_COMM_WORLD, &status);
      }


      for (ii = 1; ii <= hauteur+1; ii++)
      {
        temporairegaucheenvoie[ii]=tableauancien[ii*offset + 1];
        temporairedroiteenvoie[ii]=tableauancien[ii*offset + largeur];
      }
      
      temporairegaucheenvoie[0]=tableaunouveau[1];
      temporairedroiteenvoie[0]=tableaunouveau[largeur];
      
      if (x_grille > 0)
      {
        MPI_Isend(temporairegaucheenvoie, hauteur+2, MPI_CHAR, gauche, 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        MPI_Recv(temporairegaucherecoit, hauteur+2, MPI_CHAR, gauche, 0, MPI_COMM_WORLD, &status);
      }
      if (x_grille < largeur_grille - 1)
      {
        MPI_Isend(temporairedroiteenvoie, hauteur+2, MPI_CHAR, droite, 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
        MPI_Recv(temporairedroiterecoit, hauteur+2, MPI_CHAR, droite, 0, MPI_COMM_WORLD, &status);
      }
      
      for (ii = 1; ii <= hauteur+1; ii++)
      {
        tableauancien[ii*offset] = temporairegaucherecoit[ii];
        tableauancien[ii*offset + largeur + 1] = temporairedroiterecoit[ii];
      }

      tableaunouveau[0] = temporairegaucherecoit[0];
      tableaunouveau[largeur + 1] = temporairedroiterecoit[0];
      
      for (ii = offset; ii <= hauteur*offset; ii+=offset)
      {
        for (iii = ii+1; iii <= ii+largeur; iii++)
        {
          int voisin = tableauancien[iii-1] + tableauancien[iii+1] + tableaunouveau[iii-offset-1] + tableaunouveau[iii-offset] + tableaunouveau[iii-offset+1] + tableauancien[iii+offset-1] + tableauancien[iii+offset] + tableauancien[iii+offset+1];
          tableaunouveau[iii] = (voisin == 3 || (voisin == 2 && tableauancien[iii]));
        }   
      }
      
      if (y_grille < largeur_grille - 1)
      {
        MPI_Isend(tableaunouveau+offset*hauteur+1, largeur, MPI_CHAR, bas, 0, MPI_COMM_WORLD, &request);
        MPI_Request_free(&request);
      }
      
      //echange des deux tableaux
      char *tmp;
      tmp = tableauancien;
      tableauancien = tableaunouveau;
      tableaunouveau = tmp;
    }
    
    //je réduit localement d'abord, car on peut avoir dépassement d'entier avec un char lors de la réduction
    long densitefinale = 0;
    for (ii = offset; ii <= hauteur*offset; ii+=offset)
      for (iii = ii+1; iii <= ii+largeur; iii++)
        densitefinale += tableauancien[iii];
        
    long densitereduite;
    
    //bien que la topologie du reseau soit une grille, j'utilise MPI_Reduce par commodité, vu qu'il peut-être implémenté assez facilement (chaque ligne réduit, puis réduction entre lignes)
    MPI_Reduce(&densitefinale, &densitereduite, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 
    
    if (id == 0)
      printf("%lf\n", (double) densitereduite/taille/taille);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    free(tableau);

    free(temporaire);
  }
  else
  {
    double densitefinale = 0, densitereduite;
    MPI_Reduce(&densitefinale, &densitereduite, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    MPI_Barrier(MPI_COMM_WORLD);
  }
    
  MPI_Finalize();
	return 0;
}
