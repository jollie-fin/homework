#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
using namespace std;

int *somme;
int w = 0;
int h = 0;
int id;
int nb_proc;

#define M_Val(y) (M_Somme(somme,x2,y) - M_Somme(somme,x1-1,y))
#define M_Somme(s,x,y) (s[(y-1)+x*(h+1)]) //transformation de tableau unidimensionnel en bidimensionnel

bool sous_tab_max(int &y1f, int &y2f, int &maxf, int x1, int x2)
{
  int y1 = 1 ;
  int y1s = 1 ;
  int y2s = y1 ;
  int max = M_Val(y1);

  while(y1 <=h && M_Val(y1)<=0) //Tant que ce n'est pas positif
    {
      if(M_Val(y1) > max)
        {
          max = M_Val(y1);
          y1s= y1; //retiens la première solution
        };
      y1++ ;
    } ;
  if(y1 <= h)
    {
      int y = y1 ;
      int som = 0 ;
      max = 0 ;
      while (y <= h) //tant qu'on est dedans
        {
          while(y <= h && M_Val(y)>0) //tant que ce n'est pas négatif
            {
              som += M_Val(y); //on poursuit la solution courante
              y++ ;
            }
          if(som > max)
            {
              y2s = y-1;
              y1s = y1;
              max = som;
            }
          while(y <= h && M_Val(y)<=0) //on cherche la solution suivante
            {
              som = som + M_Val(y) ;
              y++ ;
            }
          if(som <=0)
            {
              y1s = y1 ;
              y1 = y ;
              som = 0;
              max = 0;
            }
        }
    }
  else
    {
      y2s = y1s;
    }

  if(max > maxf) //Si c'est mieux, on réactualise
    {
      y1f = y1s;
      y2f = y2s;
      maxf = max ;
      return true;
    }
  else
    {
      return false;
    }
}

void charge2(const char *nom) //fonction sans parallélisation
{
  vector<int> contenu;
  ifstream fichier(nom);
  int     h_etendu;

  w = 0;
  h = 1;
  string str;
  getline(fichier,str);
  istringstream ss(str);
  while (!ss.eof())
  {
    int tmp;
    w++;
    ss >> tmp;
    contenu.push_back(tmp);
  }
  while (!fichier.eof())
  {
    string str;
    getline(fichier,str);
    istringstream ss(str);
    for (int i = 0; i < w; i++)
    {
      int tmp;
      ss >> tmp;
      contenu.push_back(tmp);
    }
    if (ss.fail())
      break;
    h++;
  }

  somme = new int[h*(w+1)];

  for (int y = 1; y <= h; y++) //calcule la somme
  {
    M_Somme(somme,0,y) = 0;
    for (int x = 1; x <= w; x++)
    {
      M_Somme(somme,x,y) = M_Somme(somme,x-1,y) + contenu[(y-1)*w+(x-1)];
    }
  }
}


void charge(const char *nom) //solution avec parallelisation, non fonctionnelle pour cause de changement d'indice
{
  vector<int> contenu;
  ifstream fichier(nom);
  int     h_etendu;
  if (id == 0)
  {
    w = 0;
    h = 1;
    {
      string str;
      getline(fichier,str);
      istringstream ss(str);
      while (!ss.eof())
      {
        int tmp;
        w++;
        ss >> tmp;
        contenu.push_back(tmp);
      }
    }
    while (!fichier.eof())
    {
      string str;
      getline(fichier,str);
      istringstream ss(str);
      for (int i = 0; i < w; i++)
      {
        int tmp;
        ss >> tmp;
        contenu.push_back(tmp);
      }
      if (ss.fail())
        break;
      h++;
    }
    contenu.resize(w*h);

/*    cout << "/" << h << "/" << endl;
    cout << "/" << ((h+nb_proc-1)/nb_proc)*nb_proc << "/" << endl;*/


    for (int y = h; y < ((nb_proc-1+h)/nb_proc)*nb_proc; y++)
      for (int x = 0; x < w; x++)
        contenu.push_back(0);
    h_etendu = ((nb_proc-1+h)/nb_proc)*nb_proc;
  }
  
  MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h_etendu, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Comm nouveau_monde;
  MPI_Comm_split(MPI_COMM_WORLD, (id < h)?0:MPI_UNDEFINED, id, &nouveau_monde); //seule une partie des processus calculera les sommes cumulées

  if (id == 0)
  {
    somme = new int[h_etendu*(w+1)];
  }
  else
  {
    somme = new int[h*(w+1)];
  }


  if (id < h)
  {
    int h_local = h_etendu/nb_proc;
    int *a = new int[w*h_local];
    int *sommelocal = new int[(w+1)*h_local];
  
    MPI_Scatter(&contenu[0], w*h_local, MPI_INT, a, w*h_local, MPI_INT, 0, nouveau_monde);

    for (int y = 1; y <= h_local; y++)
    {
      M_Somme(sommelocal,0,y) = 0;
      for (int x = 1; x <= w; x++)
      {
        M_Somme(sommelocal,x,y) = M_Somme(sommelocal,x-1,y) + a[(y-1)*w+(x-1)];
      }
    }
    MPI_Gather(sommelocal, h_local*(w+1), MPI_INT, somme, h_local*(w+1), MPI_INT, 0, nouveau_monde);
    MPI_Comm_free(&nouveau_monde);    

    delete[] sommelocal;
    delete[] a;
  }
  MPI_Bcast(somme, (w+1)*h, MPI_INT, 0, MPI_COMM_WORLD); //operation lourde!
}



void calcule()
{
  int x1f=1;
  int x2f=1;
  int y1f=1;
  int y2f=1;
  int maxf=M_Somme(somme,1,1);
  int compteur=id;

  for (int x1 = 1; x1 <= w; x1++)
    for (int x2 = x1; x2 <= w; x2++)
      {
         compteur++;
         if (compteur == nb_proc) //ne calcule que les lignes qui lui sont attribuées
         {
           compteur = 0;
           if (sous_tab_max(y1f,y2f,maxf,x1,x2))
           {
             x1f = x1;
             x2f = x2;
           }
        }
      }

  int *x1tot, *x2tot, *y1tot, *y2tot, *maxtot;
  x1tot = new int[nb_proc];
  x2tot = new int[nb_proc];
  y1tot = new int[nb_proc];
  y2tot = new int[nb_proc];
  maxtot = new int[nb_proc];

  MPI_Gather(&x1f,1,MPI_INT,x1tot,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Gather(&x2f,1,MPI_INT,x2tot,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Gather(&y1f,1,MPI_INT,y1tot,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Gather(&y2f,1,MPI_INT,y2tot,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Gather(&maxf,1,MPI_INT,maxtot,1,MPI_INT,0,MPI_COMM_WORLD);

  if (id == 0)
  {
    for (int i = 1; i < nb_proc; i++)
    {
       if (maxtot[i] > maxf)
       {
         maxf = maxtot[i];
         x1f = x1tot[i];
         x2f = x2tot[i];
         y1f = y1tot[i];
         y2f = y2tot[i];
       }
    } //Fait une réduction 

    cout << "Meilleur tableau : (" << x1f << "," << y1f << ")/(" << x2f << "," << y2f << ")" << endl;
    cout << "Score : " << maxf << endl;
  }


  delete [] x1tot;
  delete [] x2tot;
  delete [] y1tot;
  delete [] y2tot;
  delete [] maxtot;
  delete [] somme;
}


int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
  if (argc != 2)
  {
    MPI_Finalize();
    return 1;
  }
  charge2(argv[1]);
  calcule();
  fflush(stdout);
  MPI_Finalize();
  return 0;
}
