#include "syscall.h"

int main()
{
	char a,b;
	char s[50];
	int e;
	a=GetChar();
	b=GetChar();
	printf("%d %d\n", (int) a, (int) b); //en entrant EOF, on voit que le programme renvoie systématiquement -1 dans ce cas
	GetString(s,50);
	PutString(s);
	e=GetInt();
	PutInt(e);
	printf("%.15s\n", "bonjour et bienvenue sur cette terre\n"); //Il faut préciser la taille de la chaîne à afficher avec %.XXXXs ou XXXX est la longueur
	PutChar('\n');
	PutString(
	"0001020304050607080910111213141516171819202122232425262728293031323334353637383940414243444546474849"
	"5051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899"
	" + quelques poussières\n"); //pour montrer que le buffer limité en taille n'a pas d'importance: on affiche ici plus de 200 caractères
	return 3;
}

