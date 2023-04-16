/*****************************************************************/
/*                                                               */
/* Produit de matrice par double diffusion                       */
/*                                                               */
/*****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdarg.h>

/******* Fonctions d'affichage ***********/
#define VOIRD(expr) do {printf("P%d (%d,%d) <%.3d> : \t{ " #expr " = %d }\n", \
                               my_id,i_row,i_col,__LINE__,expr);fflush(NULL);} while(0)
#define PRINT_MESSAGE(...) FPRINT_MESSAGE(my_id,i_row,i_col, __LINE__,  __VA_ARGS__)

static void FPRINT_MESSAGE(int id, int row, int col, int line, const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  printf("P%d (%d,%d) <%.3d> : \t",id,row,col,line);
  vprintf(fmt, ap);
  fflush(NULL);
  va_end(ap);
  return;
}


typedef struct { 
  int row;   /* le nombre de lignes   */
  int col;   /* le nombre de colonnes */
  double* data; /* les valeurs           */
} matrix_t;

int my_id, nb_proc;
int nb_col,nb_row;       /* Largeur et hauteur de la grille */
MPI_Comm MPI_COMM_HORIZONTAL; /* Communicateur pour diffusions horizontales */
MPI_Comm MPI_COMM_VERTICAL;   /* Communicateur pour diffusions verticales */
int i_col,i_row;         /* Position dans la grille */
int actif;

/*******************************************/
/* initialisation aléatoire de la matrice  */
/*******************************************/
void mat_init_alea(matrix_t *mat,int width, int height)
{
  int i;

  mat->row = height;
  mat->col = width;

  mat->data=(double*)calloc(height*width,sizeof(double));
  for(i=0 ; i<height*width ; i++)
    mat->data[i]=1.0*rand()/(RAND_MAX+1.0);
}

/*******************************************/
/* initialisation à 0 de la matrice        */
/*******************************************/
void mat_init_empty(matrix_t *mat,int width, int height)
{
  mat->row = height;
  mat->col = width;
  mat->data=(double*)calloc((mat->row)*(mat->col),sizeof(double));
}

/*******************************************/
/* affichage de la matrice                 */
/*******************************************/
void mat_display(matrix_t A)
{
  int i,j,t=0;

  printf("       ");
  for(j=0;j<A.col;j++)
    printf("%7d ",j);
  printf("\n");
  printf("       __");
  for(j=0;j<A.col;j++)
    printf("________");
  printf("_\n");

  for(i=0;i<A.row;i++)
    {
      printf("%6d | ",i);
      for(j=0;j<A.col;j++)
        printf("%7.3g ",A.data[t++]);
      printf("|\n");
    }
  printf("       --");
  for(j=0;j<A.col;j++)
    printf("--------");
  printf("-\n");
}

/*******************************************/
/* C+=A*B                                  */
/*******************************************/
void mat_mult(matrix_t A, matrix_t B, matrix_t C)
{
  int i,j,k,M,N,K;
  double *_A,*_B,*_C;
 
  _A=A.data;
  _B=B.data;
  _C=C.data;

  M = C.row;
  N = C.col;
  K = A.col;

  if((M!=A.row) || (N!=B.col) || (K!=B.row)) {
    PRINT_MESSAGE("Attention, tailles incompatibles");
    VOIRD(A.row);VOIRD(A.col);VOIRD(B.row);
    VOIRD(C.col);VOIRD(C.row);VOIRD(C.col);
    exit(1);
  }

  for(i=0 ; i<M ; i++)
    for(j=0 ; j<N ; j++)
      for(k=0 ; k<K ; k++) 
        _C[i*N+j]+=_A[i*K+k]*_B[k*N+j];
}

int squareroot(int taille, int n)
{
  int ret = 1, i;
  for (i = 1; i <= taille; i++)
  {
    if (i%taille == 0 && (i*i)<=n)
      ret = i;
  }
  return ret;
}

/*******************************************/
/* Initialisation de la grille             */
/*******************************************/
void init_communicateurs(int taille)
{
  int p = squareroot(taille, nb_proc);
  nb_col=p; nb_row=p;

  int col = my_id / p;
  int row = my_id % p;
  actif = (col < p);

  if (actif)
  {
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &MPI_COMM_VERTICAL);
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &MPI_COMM_HORIZONTAL);
  }
  else
  {
    MPI_Comm_split(MPI_COMM_WORLD, p, my_id, &MPI_COMM_VERTICAL);
    MPI_Comm_split(MPI_COMM_WORLD, p, my_id, &MPI_COMM_HORIZONTAL);
  }    

  MPI_Comm_rank(MPI_COMM_VERTICAL,&i_row);
  MPI_Comm_rank(MPI_COMM_HORIZONTAL,&i_col);
}

/*******************************************/
/* Produit de matrice par double diffusion */
/*******************************************/
void parallel_mat_mult(matrix_t A, matrix_t B, matrix_t C)
{
  if (!actif) return;

  matrix_t Atmp;
  mat_init_empty(&Atmp,A.row,A.col);
  matrix_t Btmp;
  mat_init_empty(&Btmp,B.row,B.col);

  int k;  

  for(k=0; k<nb_col; k++) {
    memcpy(Atmp.data,A.data,A.row*A.col*sizeof(double));
    MPI_Bcast(Atmp.data,A.row*A.col,MPI_DOUBLE, k, MPI_COMM_HORIZONTAL);
    memcpy(Btmp.data,B.data,B.row*B.col*sizeof(double));
    MPI_Bcast(Btmp.data,B.row*B.col,MPI_DOUBLE, k, MPI_COMM_VERTICAL);
    mat_mult(A,B,C);
  }
}


int main(int argc, char **argv) 
{
  int taille=0;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_id);
  
  if(argc==0)
    fprintf(stderr,"Usage : %s <taille de matrice>",argv[0]);
  else 
    taille = atoi(argv[1]);

  init_communicateurs(taille);
  if(my_id==0) {printf("%d %d %d\n",taille, nb_col, nb_row);}
  {
    matrix_t A,B,C;
    
    mat_init_alea(&A,taille/nb_col,taille/nb_row);
    mat_init_alea(&B,taille/nb_col,taille/nb_row);
    mat_init_empty(&C,taille/nb_col,taille/nb_row);
    parallel_mat_mult(A,B,C);
    if(my_id==0)
    {
      mat_display(A); mat_display(B); mat_display(C);
      mat_init_empty(&C,taille/nb_col,taille/nb_row);
      int k;
      for (k = 0; k < nb_col; k++)
        mat_mult(A,B,C);
      mat_display(C);    
    }
  }
  MPI_Finalize();
  
  return (0);
}
