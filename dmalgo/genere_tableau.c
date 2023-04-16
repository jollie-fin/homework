#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  FILE *fichier;
  int i,j;
  if (argc != 4) return 1;
  fichier = fopen(argv[3],"w");
  for (i = 0; i < atoi(argv[2]); i++)
  {
    for (j = 0; j < atoi(argv[1]); j++)
      fprintf (fichier," %d",rand()-RAND_MAX/2);
    if (i != atoi(argv[2])-1)
      fprintf(fichier,"\n");
  }
  fclose (fichier);
  return 0;
}
