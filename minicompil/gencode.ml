open Expr

let ps = print_string
let p x = print_string x;print_newline()

(*********************************************************)
(**                   Analyse statique                  **)
(*********************************************************)

(*liste indiquant les fonctions existantes*)
let presence_fonction = ref [];

exception Duplication_fonction
exception Duplication_parametres

(*construction de presence_fonction en vérifiant les duplicata*)
let rec map_nom_fonction = function
    Def(f,e1,e2,e,l) -> if List.mem f !presence_fonction
			then
			  raise Duplication_fonction
			else if e1 = e2
			then
                          raise Duplication_parametres
			else begin presence_fonction := f::!presence_fonction; map_nom_fonction l end
   |Run(e) -> ()

exception Nom_variable_inconnu
exception Nom_fonction_inconnu

(*renomme les parametres des fonction en "1" et "2"; verifie que les variables utilisees ont ete declarees*)
let rec transforme_parametre = function pr ->
  let rec change_parametre = fun nom1 nom2 e -> match e with
    Add(e1,e2) -> Add((change_parametre nom1 nom2 e1), (change_parametre nom1 nom2 e2))
   |Sub(e1,e2) -> Sub((change_parametre nom1 nom2 e1), (change_parametre nom1 nom2 e2))
   |Mul(e1,e2) -> Mul((change_parametre nom1 nom2 e1), (change_parametre nom1 nom2 e2))
   |Div(e1,e2) -> Div((change_parametre nom1 nom2 e1), (change_parametre nom1 nom2 e2))     
   |Const(k)   -> Const(k)
   |If(e0,e1,e2,e3) -> If((change_parametre nom1 nom2 e0),
			  (change_parametre nom1 nom2 e1),
			  (change_parametre nom1 nom2 e2),
			  (change_parametre nom1 nom2 e3))
   |(Var s) when s=nom1 -> Var "1"
   |(Var s) when s=nom2 -> Var "2"
   |(Var s)             -> raise Nom_variable_inconnu
   |Fon(f,e1,e2) when List.mem f !presence_fonction -> Fon(f, (change_parametre nom1 nom2 e1), (change_parametre nom1 nom2 e2))
   |Fon(_,_,_)         -> raise Nom_fonction_inconnu
 and transforme_aux = function
   |Def(f,v1,v2,e,l)   -> Def (f,"1","2",(change_parametre v1 v2 e),(transforme_aux l))
   |Run(e)             -> Run(change_parametre "" "" e)
 in
   transforme_aux pr






(****************************************************)
(**            Generation du code                  **)
(****************************************************)

(*n'est jamais levee, mais permet de filtrer tous les cas de maniere propre*)
exception Inconsistence_du_programme

(*si parpile = true, passage d'argument par pile, sinon par $a1 et $a2*)
(*convention:
    par pile: 16($fp)->var1, 12($fp)->var2, 8($fp)->ancien $fp, 4($fp)->adresse retour
    par valeur: $a1->var1, $a2->var2, 12($sp)->ancien a1, 8($sp)->ancien a2, 4($sp)->adresse retour*)
let engendre = fun pr parpile->
(*numero de la condition en cours de generation*)
  let num_if = ref 0
(*numero de l'appel de fonction en cours de generation*)
  and num_fun = ref 0
  in
  let rec engendre_fon = function
    | Var "1" -> if parpile then p "  lw $a0,16($fp) " else p "  move $a0,$a1"
    | Var "2" -> if parpile then p "  lw $a0,12($fp) " else p "  move $a0,$a2"
    | Var _ -> raise Inconsistence_du_programme
    | Fon (f,e1,e2) ->
        (*allocation d'espace sur la pile*)
	if parpile then
            p "  addiu $sp, $sp, -16"
	else
          begin
	    p "  addiu $sp, $sp, -12";  
	    p "  sw $a1, 12($sp)";
	    p "  sw $a2, 8($sp)";
	  end;

        (*evaluation de e1*)
	engendre_fon e1;

	if parpile then
  	    p "  sw $a0, 16($sp)"
	else (*on stocke provisoirement [|e1|] sur la pile a la place de l'adresse de retour*)
	    p "  sw $a0, 4($sp)";

        (*evaluation de e2*)
	engendre_fon e2;
	if parpile then
	  begin
            p "  sw $a0, 12($sp)";
	    p "  sw $fp,  8($sp)"
	  end
	else
          begin
	    p "  move $a2,$a0";
	    p "  lw $a1,4($sp)"
          end;

	p ("  la $a0,call"^(string_of_int !num_fun));
	p "  sw $a0, 4($sp)";
	
	(*le nom de la fonction est fon+nom_fonction*)
	p ("  b fon"^f);

       (*l'adresse de retour est call+num_fun*)
        p ("call"^(string_of_int !num_fun)^":");
	
	(*on restore le contexte*)
	if parpile then
	  p "  addiu $sp, $sp, 16"
        else
          begin
	    p "  lw $a1, 12($sp)";
	    p "  lw $a2, 8($sp)";
	    p "  addiu $sp, $sp, 12"
          end;
          	
	incr num_fun
	
    | Const k -> p ("  li $a0, "^(string_of_int k))
    | Add(e1,e2) -> begin
	engendre_fon e1;         (* evaluation de e1, le resultat est dans $a0 *)
	p "  sw $a0, 0($sp)";   (* on stocke le resultat en haut de la pile *)
	p "  addiu $sp, $sp-4";
	engendre_fon e2;         (* evaluation de e2, idem *)
	p "  lw $t1, 4($sp)";   (* on lit le haut de la pile, qu'on stocke dans $t1 *)
	p "  add $a0, $a0, $t1"; (* a0 <- a0 + t1 *)
	p "  addiu $sp, $sp, 4"  (* on depile *)
      end
    | Mul(e1,e2) -> begin
	engendre_fon e1;
	p "  sw $a0, 0($sp)";
	p "  addiu $sp, $sp-4";
	engendre_fon e2;
	p "  lw $t1, 4($sp)";
	p "  mul $a0, $a0, $t1";
	p "  addiu $sp, $sp, 4"	
      end
    | Sub(e1,e2) -> begin
	engendre_fon e1;
	p "  sw $a0, 0($sp)";
	p "  addiu $sp, $sp-4";
	engendre_fon e2;
	p "  lw $t1, 4($sp)";
	p "  neg $a0, $a0"; (*pour soustraire, on prend l'oppose de [|e2|]*)
	p "  add $a0, $a0, $t1";
	p "  addiu $sp, $sp, 4"	
      end
    | Div(e1,e2) -> begin
        (*si [|e2|] = 0, on saute a erreurdivision: qui s'occupe de reporter l'erreur et de quitter*)
	engendre_fon e1;
	p "  sw $a0, 0($sp)";
	p "  addiu $sp, $sp-4";
	engendre_fon e2;
	p "  lw $t1, 4($sp)";
        p "  beqz $a0, erreurdivision";
	p "  div $a0, $t1, $a0";
	p "  addiu $sp, $sp, 4"
        end
    | If (e0,e1,e2,e3) -> begin
        (*structure generale:
                si <> va a sinon+num_if
                   branche then
                   va a fin+num_if
                sinon+num_if:
                   branche else
                fin+num_if*)
	let n = !num_if in (*on retient le numero de la conditionnelle, si jamais des conditionnelles sont imbriquees*)
	incr num_if;
	engendre_fon e0;
	p "  sw $a0, 0($sp)";
	p "  addiu $sp, $sp-4";
	engendre_fon e1;
	p "  lw $t1, 4($sp)";
	p "  addiu $sp, $sp, 4";
        p ("  bne $a0,$t1,else"^(string_of_int n));
        engendre_fon e2;
        p ("  b end"^(string_of_int n));
        p ("else"^(string_of_int n)^":");
        engendre_fon e3;
        p ("end"^(string_of_int n)^":");
        end
  and
     engendre_prog = function pr -> match pr with
     Def(f,_,_,e,l) ->
	p ("fon"^f^":");
	(*recupere (eventuellement) le frame pointer*)
        if parpile then
	  p ("  move $fp,$sp");
	engendre_fon e;
	(*adresse de retour*)
	p "  lw $t1, 4($sp)";
	(*restoration (eventuelle) du frame pointer*)
	if parpile then	
	  p "  lw $fp, 8($sp)";
	(*retour*)
	p "  jr $t1";	
	
	engendre_prog l
    |Run(e) ->
	p ("main:");
	(*initialisation du frame pointer*)
	p ("  move $fp,$sp");
	engendre_fon e;
  in
    (*analyse statique*)
    map_nom_fonction pr;
    let nouveau_pr = transforme_parametre pr
    in
        (*generation du code*)
        engendre_prog nouveau_pr;

