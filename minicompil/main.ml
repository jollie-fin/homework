open Expr
open Gencode

(* le d�but du fichier SPIM *)
let en_tete () =
  begin
    p ".data";    (* partie des donnees *)
    p "str: .asciiz \" est le resultat de l'evaluation -------------\\n\"";
    p "division_by_zero: .asciiz \" erreur : division par zero\\n\" ";
    p ".text";    (* code du programme: il faut d�finir le label main *)
    p ".globl main";
    (*traitement de l'exception division par zero*)
    p "erreurdivision:";
    p "  li $v0, 4";
    p "  la $a0, division_by_zero";
    p "  syscall";
    p "  li $v0, 10  # 10 est le code de `exit'";   
    p "  syscall"
  end


(* le code � la fin, pour afficher le resultat *)
let concl () =
  begin
    p "  li $v0, 1   # 1 pour afficher un entier";
    p "  syscall";
    p "  li $v0, 4    # 4 pour afficher une chaine";
    p "  la $a0, str";
    p "  syscall";
    p "  li $v0, 10  # 10 est le code de `exit'";   
    p "  syscall"
  end


let compile pr =
  begin
    en_tete ();
    engendre pr true; (*true:appel par pile false:appel par valeur*)
    concl ();
  end

(* stdin d�signe l'entr�e standard (le clavier) *)
(* lexbuf est un canal ouvert sur stdin *)

let lexbuf = Lexing.from_channel stdin


(* on encha�ne les tuyaux: lexbuf est pass� � Lexer.token,
   et le r�sultat est donn� � Parser.main *)

let parse () = Parser.main Lexer.token lexbuf


(* la boucle interactive *)
let calc () =
  compile (parse ());flush stdout;; (*pour mieux diagnostiquer les erreurs, j'ai enlev� le rattrapage d'exception*)

calc()
