#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define f(regle,tableau,x,y,z) (0 != (regle & (1<<((tableau[x]<<2)|(tableau[y]<<1)|tableau[z]))))

int main(int argc, char **argv)
{
  if (argc != 4)
    return 1;
  int nbetapes = atoi(argv[2]);
  int regle = atoi(argv[1]);
  int taille = strlen(argv[3]);
  //allocation
  char *tableau = (char *) calloc(taille*2, sizeof(char));
  
  char *tableauancien = tableau;
  char *tableaunouveau = tableau+taille;
  int i,ii;
  
  //initialisation du tableau
  for (i = 0; i < taille; i++)
    tableauancien[i] = (argv[3][i] == '1');
  
  //calcul effectif
  for (i = 0; i < nbetapes; i++)
  {
    for (ii = 1; ii < taille-1; ii++)
      tableaunouveau[ii] = f(regle,tableauancien, ii-1, ii, ii+1);
    tableaunouveau[0] = f(regle,tableauancien, taille-1, 0, 1);
    tableaunouveau[taille-1] = f(regle,tableauancien, taille-2, taille-1, 0);
    //echange des deux tableaux
    char *tmp;
    tmp = tableauancien;
    tableauancien = tableaunouveau;
    tableaunouveau = tmp;
  }
  
  for (i = 0; i < taille; i++)
    printf("%d",tableauancien[i]);
  printf("\n");
  free(tableau);
	return 0;
}
