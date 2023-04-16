{
open Parser;;        (* le type "token" est défini dans parser.mli *)
exception Eof;;
}

rule token = parse    (* la "fonction" aussi s'appelle token .. *)
  | [' ' '\t']     { token lexbuf }    (* on saute les blancs *)
                                       (* token: appel récursif *)
                                       (* lexbuf: argument implicite
                                          associé au tampon où sont
                                          lus les caractères *)
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
  | ['a'-'z''A'-'Z''_']['a'-'z''A'-'Z''_''0'-'9']* as s { ID s } (* un identifiant commence par une lettre ou _ et continue par une succession de caractères alphanumériques*)
  | eof             { raise Eof } 
