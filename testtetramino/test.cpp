#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;

unsigned long position[1200][2];
int nb_position = 0;

unsigned long tableau[2];

void rotationx(int pieces[5][3])
{
	for(int i = 0; i < 5; i++)
	{
		int tmp = pieces[i][1];
		pieces[i][1] = pieces[i][2];
		pieces[i][2] = -tmp;
	}
}

void rotationy(int pieces[5][3])
{
	for(int i = 0; i < 5; i++)
	{
		int tmp = pieces[i][0];
		pieces[i][0] = pieces[i][2];
		pieces[i][2] = -tmp;
	}
}

void rotationz(int pieces[5][3])
{
	for(int i = 0; i < 5; i++)
	{
		int tmp = pieces[i][1];
		pieces[i][1] = pieces[i][0];
		pieces[i][0] = -tmp;
	}
}

void translationx(int pieces[5][3], int valeur)
{
	for(int i = 0; i < 5; i++)
		pieces[i][0] += valeur;
}

void translationy(int pieces[5][3], int valeur)
{
	for(int i = 0; i < 5; i++)
		pieces[i][1] += valeur;
}

void translationz(int pieces[5][3], int valeur)
{
	for(int i = 0; i < 5; i++)
		pieces[i][2] += valeur;
}

void insere_si_pas_dedans(int pieces[5][3])
{
	unsigned long hash[2] = {0,0};
	for (int i = 0; i < 5; i++)
		if (pieces[i][0] < 0 || pieces[i][1] < 0 || pieces[i][2] < 0 || pieces[i][0] > 4 || pieces[i][1] > 4 || pieces[i][2] > 4)
			return;

	for (int i = 0; i < 5; i++)
	{
		long p = pieces[i][0] + 5 * pieces[i][1] + 25 * pieces[i][2];
		if (p >= 125)
		{
			cout << "hahahah" << endl;
			return;
		}		
		if (p >= 64)
		{
			hash[1] |= (1UL << (p-64));
		}
		else
		{
			hash[0] |= (1UL << p);
		}
	}

	for (int i = 0; i < nb_position; i++)
	{
		if (position[i][0] == hash[0] && position[i][1] == hash[1])
			return;
	}
	position[nb_position][0] = hash[0];
	position[nb_position][1] = hash[1];

	nb_position++;
}

void toutes_translations(int pieces[5][3])
{
	translationx(pieces, -5);
	for (int ix = 0; ix < 10; ix++)
	{
		translationx(pieces, 1);
		translationy(pieces, -5);
		for (int iy = 0; iy < 10; iy++)
		{
			translationy(pieces, 1);	
			translationz(pieces, -5);
			for (int iz = 0; iz < 10; iz++)
			{
				translationz(pieces, 1);
				insere_si_pas_dedans(pieces);
			}
			translationz(pieces, -5);
		}
		translationy(pieces, -5);
	}
	translationx(pieces, -5);
}

void initialise()
{
	int pieces[5][3] = {{0,0,0},{0,0,1},{0,0,2},{0,0,3},{0,1,1}};
	for (int i = 0; i < 1200; i++)
		position[i][0] = position[i][1] = 0;
	nb_position = 0;
	tableau[0] = 0UL;
	tableau[1] = 0UL;
	for (int i = 0; i < 4; i++)
	{
		rotationx(pieces);
		toutes_translations(pieces);
	}
	rotationy(pieces);
	for (int i = 0; i < 4; i++)
	{
		rotationz(pieces);
		toutes_translations(pieces);
	}
	rotationy(pieces);
	for (int i = 0; i < 4; i++)
	{
		rotationx(pieces);
		toutes_translations(pieces);
	}
	rotationy(pieces);
	for (int i = 0; i < 4; i++)
	{
		rotationz(pieces);
		toutes_translations(pieces);
	}
	rotationy(pieces);
	rotationz(pieces);
	for (int i = 0; i < 4; i++)
	{
		rotationy(pieces);
		toutes_translations(pieces);
	}
	rotationz(pieces);
	rotationz(pieces);
	for (int i = 0; i < 4; i++)
	{
		rotationy(pieces);
		toutes_translations(pieces);
	}
	rotationz(pieces);
}

bool test(int profondeur)
{
	static long best = 0UL;
	for (int i = 0; i < nb_position; i++)
	{
		if (((tableau[0] & position[i][0]) == 0UL) && ((tableau[1] & position[i][1]) == 0UL))
		{
			if (profondeur == 25)
			{
				cout << i << endl;
				return true;
			}
			else
			{
				tableau[0]|=position[i][0]; tableau[1]|=position[i][1];
				if (test(profondeur+1))
				{
					cout << i << endl;
					return true;
				}
				else
				{
					tableau[0]&= ~position[i][0];				
					tableau[1]&= ~position[i][1];
				}
			}
		}
	}
	return false;
}
int main()
{
	initialise();
	cout << test(5) << endl;
return 0;
}
