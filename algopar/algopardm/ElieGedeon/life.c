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
  int largeur_grille_processeur = 1;
  while ((largeur_grille_processeur+1) * (largeur_grille_processeur+1) <= tot)
    largeur_grille_processeur++;
  largeur_grille_processeur = MIN(largeur_grille_processeur, taille);

  int actif  = id < largeur_grille_processeur*largeur_grille_processeur;
  
  if (actif)
  {
    int taillelocale = (taille+largeur_grille_processeur-1)/largeur_grille_processeur;
    
    int gauche = (id%largeur_grille_processeur != 0)              ? id-1              : id+largeur_grille_processeur-1;
    int droite = (id%largeur_grille_processeur != largeur_grille_processeur-1) ? id+1              : id-largeur_grille_processeur+1;
    int haut   = (id/largeur_grille_processeur != 0)              ? id - largeur_grille_processeur : id + largeur_grille_processeur*(largeur_grille_processeur-1);
    int bas    = (id/largeur_grille_processeur != largeur_grille_processeur-1) ? id + largeur_grille_processeur : id - largeur_grille_processeur*(largeur_grille_processeur-1);

    int largeur = MIN(taillelocale, taille-(id%largeur_grille_processeur)*taillelocale);
    int hauteur = MIN(taillelocale, taille-(id/largeur_grille_processeur)*taillelocale);

    //printf ("%d: actif, grille de taille %dx%d\n", id, largeur, hauteur);
    
    int offset = largeur+2;
    
    //allocation. On alloue plus grand, car on inclut les cases extremes des voisins
    char *tableau = (char *) calloc(2*offset*(hauteur+2), sizeof(char));
    char *tableauancien = tableau;
    char *tableaunouveau = tableau + offset*(hauteur+2);
    char *temporaire = (char *) calloc(4*hauteur, sizeof(char));
    
    char *temporairegaucheenvoie = temporaire;
    char *temporairedroiteenvoie = temporaire+hauteur;
    char *temporairegaucherecoit = temporaire+hauteur * 2;
    char *temporairedroiterecoit = temporaire+hauteur * 3;
    
    int i,ii,iii;
    
    srand(time(NULL)+id);
    
    for (ii = 1*offset; ii <= hauteur*offset; ii+=offset)
      for (iii = ii+1; iii <= ii+largeur; iii++)
        tableauancien[iii] = (((double) rand()) / ((double) RAND_MAX)) < densite;
    
    for (i = 0; i < nbetapes; i++)
    {
      for (ii = 0; ii < hauteur; ii++)
      {
        temporairegaucheenvoie[ii]=tableauancien[(ii+1)*offset + 1];
        temporairedroiteenvoie[ii]=tableauancien[(ii+1)*offset + largeur];
      }
      
      MPI_Request request1;
      MPI_Request request2;
      MPI_Status status;
      
      MPI_Isend(temporairegaucheenvoie, hauteur, MPI_CHAR, gauche, 0, MPI_COMM_WORLD, &request1);
      MPI_Isend(temporairedroiteenvoie, hauteur, MPI_CHAR, droite, 0, MPI_COMM_WORLD, &request2);
      MPI_Recv(temporairegaucherecoit, hauteur, MPI_CHAR, gauche, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(temporairedroiterecoit, hauteur, MPI_CHAR, droite, 0, MPI_COMM_WORLD, &status);
    
      for (ii = 0; ii < hauteur; ii++)
      {
        tableauancien[(ii+1)*offset] = temporairegaucherecoit[ii];
        tableauancien[(ii+1)*offset + largeur + 1] = temporairedroiterecoit[ii];
      }
      
      MPI_Isend(tableauancien+offset, largeur+2, MPI_CHAR, haut, 0, MPI_COMM_WORLD, &request1);
      MPI_Isend(tableauancien+offset*hauteur, largeur+2, MPI_CHAR, bas, 0, MPI_COMM_WORLD, &request2);
      MPI_Recv(tableauancien, largeur+2, MPI_CHAR, haut, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(tableauancien+offset*(hauteur+1), largeur+2, MPI_CHAR, bas, 0, MPI_COMM_WORLD, &status);
    
      for (ii = offset; ii <= hauteur*offset; ii+=offset)
      {
        for (iii = ii+1; iii <= ii+largeur; iii++)
        {
          int voisin = tableauancien[iii-1] + tableauancien[iii+1] + tableauancien[iii-offset-1] + tableauancien[iii-offset] + tableauancien[iii-offset+1] + tableauancien[iii+offset-1] + tableauancien[iii+offset] + tableauancien[iii+offset+1];
          tableaunouveau[iii] = (voisin == 3 || (voisin == 2 && tableauancien[iii]));
        }   
      }
      
      //echange des deux tableaux
      char *tmp;
      tmp = tableauancien;
      tableauancien = tableaunouveau;
      tableaunouveau = tmp;
    }
    
    //je réduit localement d'abord, car on peut avoir dépassement d'entier avec un char lors de la réduction
    double densitefinale = 0;
    for (ii = offset; ii <= hauteur*offset; ii+=offset)
      for (iii = ii+1; iii <= ii+largeur; iii++)
        densitefinale += tableauancien[iii];
    densitefinale /= largeur*hauteur;
    
    double densitereduite;
    
    //bien que la topologie du reseau soit une grille, j'utilise MPI_Reduce par commodité, vu qu'il peut-être implémenté assez facilement (chaque ligne réduit, puis réduction entre lignes)
    MPI_Reduce(&densitefinale, &densitereduite, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
    
    if (id == 0)
      printf("%lf\n", (double) densitereduite/largeur_grille_processeur/largeur_grille_processeur);
    
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
