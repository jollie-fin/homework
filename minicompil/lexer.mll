{
open Parser;;        (* le type "token" est d�fini dans parser.mli *)
exception Eof;;
}

rule token = parse    (* la "fonction" aussi s'appelle token .. *)
  | [' ' '\t']     { token lexbuf }    (* on saute les blancs *)
                                       (* token: appel r�cursif *)
                                       (* lexbuf: argument implicite
                                          associ� au tampon o� sont
                                          lus les caract�res *)
  | '\n'            { EOL }
  | ['0'-'9']+ as k { INT (int_of_string k) }
  | '+'             { PLUS }
  | '*'             { TIMES }
  | '('             { LPAREN }
  | ')'             { RPAREN }
  | '/'             { DIVIDE }
  | '-'             { MINUS }
  | "if"            { IF }
  | '='             { EQUAL }
  | "then"          { THEN }
  | "else"          { ELSE }
  | "endif"         { FI }
  | "def"           { DEF }
  | ';'             { POINT_VIRGULE }
  | ','             { VIRGULE }
  | "run"           { RUN }
  | '.'             { POINT }
  | ['a'-'z''A'-'Z''_']['a'-'z''A'-'Z''_''0'-'9']* as s { ID s } (* un identifiant commence par une lettre ou _ et continue par une succession de caract�res alphanum�riques*)
  | eof             { raise Eof } 
