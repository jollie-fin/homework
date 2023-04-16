(* un type pour des expressions arithmétiques simples *)
type expr =
    Const of int
  | Add of expr*expr
  | Mul of expr*expr
  | Sub of expr*expr
  | Div of expr*expr
  | If of expr*expr*expr*expr (*if e0 = e1 then e2 else e3*)
  | Var of string
  | Fon of string*expr*expr (*appel de fonctions*)


type programme =
   Def of string*string*string*expr*programme (* nom_fonction, nom_parametre1, nom_parametre2, corps, suite du programme *)
  |Run of expr

 (* fonction d'affichage *)
let rec affiche_prog pr = 
  let rec affiche_expr e =
    let aff_aux s a b = 
        begin
          print_string s;
          affiche_expr a;
          print_string ", ";
          affiche_expr b;
          print_string ")"
        end
    in
    match e with
    | Const k -> print_int k
    | Add(e1,e2) -> aff_aux "Add(" e1 e2
    | Mul(e1,e2) -> aff_aux "Mul(" e1 e2
    | Sub(e1,e2) -> aff_aux "Sub(" e1 e2
    | Div(e1,e2) -> aff_aux "Div(" e1 e2
    | If(e0,e1,e2,e3) ->
          begin
            print_string "If(";
            affiche_expr e0;
            print_string ",";
            affiche_expr e1;
            aff_aux "," e2 e3;
          end
    | Fon(f,e1,e2) -> aff_aux (f^"(") e1 e2
    | Var v -> print_string v
  in
    match pr with
      Run(e) -> print_string "Main:"; affiche_expr e; print_newline()
    | Def(f,x,y,e,l) -> print_string (f^":("^x^","^y^") |--> "); affiche_expr e; print_newline(); affiche_prog l

(*TODO: implémenter eval avec les appels de fonctions. Non nécessaire au bon fonctionnement du compilateur*)
exception Division_by_zero
exception Not_yet_implemented
let rec eval = function
  | Const k -> k
  | Add(e1,e2) -> (eval e1) + (eval e2)
  | Mul(e1,e2) -> (eval e1) * (eval e2)
  | Sub(e1,e2) -> (eval e1) - (eval e2)
  | Div(e1,e2) -> let diviseur = eval e2 in
                     if diviseur == 0 then raise Division_by_zero
                     else eval e1 / diviseur
  | If(e0,e1,e2,e3) -> if (eval e0) = (eval e1) then eval e2 else eval e3
  |_ -> raise Not_yet_implemented
