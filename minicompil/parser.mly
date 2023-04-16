%{
(* --- pr�ambule: ici du code Caml --- *)

open Expr

%}
/* description des lex�mes */

%token <int> INT       /* le lex�me INT a un attribut entier */
%token <string> ID
%token PLUS TIMES MINUS DIVIDE
%token DEF POINT_VIRGULE VIRGULE RUN POINT
%token LPAREN RPAREN
%token VAR
%token IF EQUAL THEN ELSE FI
%token EOL             /* retour � la ligne */

/* pas d'ambiguite sur le if then else, donc pas besoin de notion de pr�c�dence */
%left PLUS MINUS             /* associativit� + pr�c�dence minimale */
%left TIMES DIVIDE            /* pr�c�dence maximale */


%start main             /* "start" signale le point d'entr�e */
%type <Expr.programme> main     /* on _doit_ donner le type du point d'entr�e */

%%
    /* --- d�but des r�gles de grammaire --- */

main:                       /* � droite, les valeurs associ�es */
    programme EOL                { $1 }
;

programme:
  | DEF ID LPAREN ID VIRGULE ID RPAREN EQUAL expr POINT_VIRGULE programme {Def ($2,$4,$6,$9,$11)}
  | RUN expr POINT {Run $2}
;

expr:                            
  | INT                     { Const $1 }
  | ID LPAREN expr VIRGULE expr RPAREN { Fon($1,$3,$5) }
  | IF expr EQUAL expr THEN expr ELSE expr FI { If($2,$4,$6,$8) }
  | ID                 { Var $1 }
  | LPAREN expr RPAREN      { $2 }
  | expr PLUS expr          { Add($1,$3) }
  | expr TIMES expr         { Mul($1,$3) }
  | expr MINUS expr          { Sub($1,$3) }
  | expr DIVIDE expr         { Div($1,$3) }
;






