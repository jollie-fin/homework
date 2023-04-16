all : run-ca-1d-cycle run-ca-1d-unbounded life deplife xlife
	

run-ca-1d-cycle : question1.c
	gcc -Wall -Wextra question1.c -o run-ca-1d-cycle

run-ca-1d-unbounded : question2.c
	mpicc -Wall -Wextra question2.c -o run-ca-1d-unbounded

life : life.c
	mpicc -Wall -Wextra life.c -o life

deplife : deplife.c
	mpicc -Wall -Wextra deplife.c -o deplife

xlife : xlife.c
	mpicc -Wall -Wextra xlife.c -o xlife

clean : 
	rm xlife deplife life run-ca-1d-unbounded run-ca-1d-cycle

