#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))

#define LARGEUR 8192
#define HAUTEUR 8192

#define TAG(x,y)((x)+LARGEUR*(y))

#define X(t)((t)%LARGEUR-LARGEUR/2)
#define Y(t)((t)/LARGEUR-LARGEUR/2)
#define HAUT(T)((T)-LARGEUR)
#define BAS(T)((T)+LARGEUR)
#define GAUCHE(T)((T)-1)
#define DROITE(T)((T)+1)


//structure de type tableau extensible
typedef struct
{
  int taille;
  int allocation;
  int *tags;
} candidat;

candidat Candidat_Cree()
{
  candidat retour;
  retour.taille = 0;
  retour.allocation = 10;
  retour.tags = malloc(10*sizeof(int));
  return retour;
}

void Candidat_Insere(candidat *c, int tag)
{
  int i;
  
  for (i = 0; i < c->taille; i++)
    if (c->tags[i] == tag)
      return;
  if (c->taille >= c->allocation)
  {
    c->allocation *= 2;
    c->tags = realloc(c->tags, c->allocation);
  }
  c->tags[c->taille] = tag;
  c->taille++;
}

void Candidat_Supprime(candidat c)
{
  free(c.tags);
}


//liste chainee grille locale

typedef struct tableau_virtuel tableau_virtuel;

struct tableau_virtuel
{
  int gauche, droite, haut, bas;
  char *zone;
  int tag;
  char *ancien;
  char *nouveau;
  tableau_virtuel *suivant;
};

tableau_virtuel *recherche(tableau_virtuel *t, int tag)
{
  if (!t) return NULL;
  if (t->tag == tag) return t;
  return recherche(t->suivant, tag);
}

tableau_virtuel *cree(int largeur, int tag, tableau_virtuel *suivant)
{
  tableau_virtuel *retour;
  retour = (tableau_virtuel *) malloc(sizeof(tableau_virtuel));
  retour->gauche = retour->droite = retour->haut = retour->bas = 0;
  retour->suivant = suivant;
  retour->tag = tag;
  retour->zone = (char *) calloc((largeur+2)*(largeur+2)*2, sizeof(char));
  retour->ancien = retour->zone;
  retour->nouveau = retour->zone + (largeur+2)*(largeur+2);
  return retour;
}

void affiche(int no, int id, tableau_virtuel *suivant)
{
  if (!suivant) return;
  printf ("%d:%dx%d n%d %d %d %d %d %d\n", id, X(suivant->tag), Y(suivant->tag), no, suivant->ancien[4], !!suivant->haut, !!suivant->bas, !!suivant->gauche, !!suivant->droite);
  affiche(no, id, suivant->suivant);
}

int main(int argc, char **argv)
{
  
	int id, tot;

  MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &tot);

  if (argc != 3)
    MPI_Abort(MPI_COMM_WORLD,1);

  //initialisation
  int taille = 100;
 
  int largeur_grille = 1;

  while ((largeur_grille+1) * (largeur_grille+1) <= tot)
    largeur_grille++;

  int col = id%largeur_grille;
  int lig = id/largeur_grille;
    
  int nbetapes = atoi(argv[1]);
    
  int actif  = id < largeur_grille*largeur_grille;
  
  int largeur = taille;
  int hauteur = taille;

  int offset = largeur+2;
  
  int i,ii,iii;      
  tableau_virtuel *t;
 
  tableau_virtuel *liste_tableau = NULL;
  if (actif)
  {
    //lecture fichier
    char *buffer = calloc(taille+1, sizeof(char));
    FILE *fichier;
    fichier = fopen(argv[2], "r");
    if (!fichier)
      MPI_Abort(MPI_COMM_WORLD,2);
    int w,h;
    fscanf(fichier,"%d", &w);
    fscanf(fichier,"%d", &h);
    int wp = (w+taille-1)/taille;
    int hp = (h+taille-1)/taille;
    
    int j,jj;
    
    for (j = -1; j < hp+1; j++)
    {
      for (jj = -1; jj < wp+1; jj++)
      {
        if (((jj+LARGEUR/2) % largeur_grille == col) && ((j+HAUTEUR/2) % largeur_grille == lig))
        {
          int tag = TAG(jj+LARGEUR/2,j+HAUTEUR/2);
          liste_tableau = cree(taille, tag, liste_tableau);
          if (j > -1) liste_tableau->haut = HAUT(tag);
          if (jj > -1) liste_tableau->gauche = GAUCHE(tag);
          if (j < hp) liste_tableau->bas = BAS(tag);
          if (jj < wp) liste_tableau->droite = DROITE(tag);
        }
      }
    }
    
    for (j = 0; j < hp; j++)
    {
      for (i = 0; i < taille; i++)
      {
        for (jj = 0; jj < wp; jj++)
        {
          memset(buffer, 0, taille);
          do
          {
            fgets (buffer, taille+1, fichier);
          } while (buffer[0] != '0' && buffer[0] != '1');
            
          for (ii = 0; ii < taille; ii++)
            buffer[ii] = (buffer[ii] == '1')?1:0;
          
          if (((jj+LARGEUR/2) % largeur_grille == col) && ((j+HAUTEUR/2) % largeur_grille == lig))
          {
            
            tableau_virtuel *tab = recherche(liste_tableau, TAG(jj+LARGEUR/2,j+HAUTEUR/2));
            memcpy(tab->ancien+offset*(i+1)+1, buffer, taille);
          }
        }
      }
    }
    
    fclose (fichier);
    free(buffer);
 

    int gauche = (col != 0)              ? id-1              : id+largeur_grille-1;
    int droite = (col != largeur_grille-1) ? id+1              : id-largeur_grille+1;
    int haut   = (lig != 0)              ? id - largeur_grille : id + largeur_grille*(largeur_grille-1);
    int bas    = (lig != largeur_grille-1) ? id + largeur_grille : id - largeur_grille*(largeur_grille-1);

    char *temporaire = (char *) calloc(4*hauteur, sizeof(char));
    
    char *temporairegaucheenvoie = temporaire;
    char *temporairedroiteenvoie = temporaire+hauteur;
    char *temporairegaucherecoit = temporaire+hauteur * 2;
    char *temporairedroiterecoit = temporaire+hauteur * 3;
    
 
    MPI_Request request;
  
      
    for (i = 0; i < nbetapes; i++)
    {
      //reinitialisation des listes des candidats au dépassements
      candidat cand[4];
      cand[0] = Candidat_Cree();
      cand[1] = Candidat_Cree();
      cand[2] = Candidat_Cree();
      cand[3] = Candidat_Cree();

      //echange latéraux
      for (t = liste_tableau; t != NULL; t = t->suivant)
      {
        if (t->gauche)
        {
          for (ii = 0; ii < hauteur; ii++)
            temporairegaucheenvoie[ii]=t->ancien[(ii+1)*offset + 1];
          MPI_Isend(temporairegaucheenvoie, hauteur, MPI_CHAR, gauche, t->gauche, MPI_COMM_WORLD, &request);
          MPI_Request_free(&request);
        }
        
        if (t->droite)
        {
          for (ii = 0; ii < hauteur; ii++)
            temporairedroiteenvoie[ii]=t->ancien[(ii+1)*offset + largeur];
          MPI_Isend(temporairedroiteenvoie, hauteur, MPI_CHAR, droite, t->droite, MPI_COMM_WORLD, &request);
          MPI_Request_free(&request);
          }
      }

      for (t = liste_tableau; t != NULL; t = t->suivant)
      {
        
        if (t->gauche)
        {
          MPI_Recv(temporairegaucherecoit, hauteur, MPI_CHAR, gauche, t->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (ii = 0; ii < hauteur; ii++)
            t->ancien[(ii+1)*offset] = temporairegaucherecoit[ii];
        }
        
        if (t->droite)
        {
          MPI_Recv(temporairedroiterecoit, hauteur, MPI_CHAR, droite, t->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (ii = 0; ii < hauteur; ii++)
            t->ancien[(ii+1)*offset+largeur+1] = temporairedroiterecoit[ii];
        }
        
      }

      //échange haut bas
      for (t = liste_tableau; t != NULL; t = t->suivant)
      {
        if (t->haut)
        {
          MPI_Isend(t->ancien+offset, largeur+2, MPI_CHAR, haut, t->haut, MPI_COMM_WORLD, &request);
          MPI_Request_free(&request);
        }
        
        if (t->bas)
        {
          MPI_Isend(t->ancien+offset*hauteur, largeur+2, MPI_CHAR, bas, t->bas, MPI_COMM_WORLD, &request);
          MPI_Request_free(&request);
        }
      }

      for (t = liste_tableau; t != NULL; t = t->suivant)
      {
        if (t->haut)
          MPI_Recv(t->ancien, largeur+2, MPI_CHAR, haut, t->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (t->bas)
          MPI_Recv(t->ancien+offset*(hauteur+1), largeur+2, MPI_CHAR, bas, t->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      
      //calcul effectif
      for (t = liste_tableau; t != NULL; t = t->suivant)
      {
        for (ii = offset; ii <= hauteur*offset; ii+=offset)
        {
          for (iii = ii+1; iii <= ii+largeur; iii++)
          {
            //si on touche le bord, on rajoute au reste
            int voisin = t->ancien[iii-1] + t->ancien[iii+1] + t->ancien[iii-offset-1] + t->ancien[iii-offset] + t->ancien[iii-offset+1] + t->ancien[iii+offset-1] + t->ancien[iii+offset] + t->ancien[iii+offset+1];
            if ((t->nouveau[iii] = (voisin == 3 || (voisin == 2 && t->ancien[iii]))))
            {
              if (ii==offset)
              {
                Candidat_Insere(cand+0, t->tag);
                t->haut=HAUT(t->tag);
                
              }
              if (ii==hauteur*offset)
              {
                Candidat_Insere(cand+1, t->tag);
                t->bas=BAS(t->tag);
                                
              }
              if (iii==ii+1)
              {
                Candidat_Insere(cand+2, t->tag);
                t->gauche=GAUCHE(t->tag);
              }
              if (iii==ii+largeur)
              {
                Candidat_Insere(cand+3, t->tag);
                t->droite=DROITE(t->tag);
                                
              }
            }
          }   
        }
  
        //echange des deux tableaux
        char *tmp;
        tmp = t->ancien;
        t->ancien = t->nouveau;
        t->nouveau = tmp;
      }
  
      //echange des dépassements
      MPI_Isend(&(cand[0].taille), 1, MPI_INT, haut, 0, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(&(cand[1].taille), 1, MPI_INT, bas, 0, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(&(cand[2].taille), 1, MPI_INT, gauche, 0, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(&(cand[3].taille), 1, MPI_INT, droite, 0, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      
      int taille_cand[4];
      
      MPI_Recv(taille_cand+0, 1, MPI_INT, haut, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(taille_cand+1, 1, MPI_INT, bas, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(taille_cand+2, 1, MPI_INT, gauche, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(taille_cand+3, 1, MPI_INT, droite, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int *tags[4];
      tags[0] = (int *) calloc(taille_cand[0], sizeof(int));
      tags[1] = (int *) calloc(taille_cand[1], sizeof(int));
      tags[2] = (int *) calloc(taille_cand[2], sizeof(int));
      tags[3] = (int *) calloc(taille_cand[3], sizeof(int));
      
      MPI_Isend(cand[0].tags, cand[0].taille, MPI_INT, haut, 1, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(cand[1].tags, cand[1].taille, MPI_INT, bas, 1, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(cand[2].tags, cand[2].taille, MPI_INT, gauche, 1, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      MPI_Isend(cand[3].tags, cand[3].taille, MPI_INT, droite, 1, MPI_COMM_WORLD, &request);
      MPI_Request_free(&request);
      
      MPI_Recv(tags[0], taille_cand[0], MPI_INT, haut, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(tags[1], taille_cand[1], MPI_INT, bas, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(tags[2], taille_cand[2], MPI_INT, gauche, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(tags[3], taille_cand[3], MPI_INT, droite, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      Candidat_Supprime(cand[0]);
      Candidat_Supprime(cand[1]);
      Candidat_Supprime(cand[2]);
      Candidat_Supprime(cand[3]);
      
      //Création des parties manquantes pour le dépassement
      for (ii = 0; ii < taille_cand[0]; ii++)
      {
        t = recherche(liste_tableau,BAS(tags[0][ii]));
        if (t == NULL)
        {
          liste_tableau = cree(taille, BAS(tags[0][ii]), liste_tableau);
          liste_tableau->haut = tags[0][ii];
        }
        else
        {
          t->haut = tags[0][ii];
        }
      }
      
      for (ii = 0; ii < taille_cand[1]; ii++)
      {
        t = recherche(liste_tableau,HAUT(tags[1][ii]));
        if (t == NULL)
        {
          liste_tableau = cree(taille, HAUT(tags[1][ii]), liste_tableau);
          liste_tableau->bas = tags[1][ii];
        }
        else
        {
          t->bas = tags[1][ii];
        }
      }

      for (ii = 0; ii < taille_cand[2]; ii++)
      {
        t = recherche(liste_tableau,DROITE(tags[2][ii]));
        if (t == NULL)
        {
          liste_tableau = cree(taille, DROITE(tags[2][ii]), liste_tableau);
          liste_tableau->gauche = tags[2][ii];
        }
        else
        {
          t->gauche = tags[2][ii];
        }
      }

      for (ii = 0; ii < taille_cand[3]; ii++)
      {
        t = recherche(liste_tableau,GAUCHE(tags[3][ii]));
        if (t == NULL)
        {
          liste_tableau = cree(taille, GAUCHE(tags[3][ii]), liste_tableau);
          liste_tableau->droite = tags[3][ii];
        }
        else
        {
          t->droite = tags[3][ii];
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      
      free(tags[0]);
      free(tags[1]);
      free(tags[2]);
      free(tags[3]);
    }
  }
  else
  {
    for (i = 0; i < nbetapes; i++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  
  //reduction
  long resultat = 0;

  for (t = liste_tableau; t != NULL; t = t->suivant)
  {
    for (i = offset; i <= offset*hauteur; i+=offset)
    {
      for (ii = i+1; ii <= i+largeur; ii++)
      {
        resultat += t->ancien[ii];
      }
    }
  }
  
  long resultatreduit;
  MPI_Reduce(&resultat, &resultatreduit, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (id == 0)
    printf ("%ld\n", resultatreduit);
  
  MPI_Finalize();
	return 0;
}


        