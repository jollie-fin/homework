#ifdef CHANGED
#include "copyright.h"
#include "system.h"
#include "synchconsole.h"
#include "synch.h"
#include <cstdio>

#include <iostream>
static Semaphore *getString;
static Semaphore *readAvail;
static Semaphore *writeDone;
static Semaphore *putString;

static void ReadAvail(void *arg) { (void) arg; readAvail->V(); }
static void WriteDone(void *arg) { (void) arg; writeDone->V(); }
SynchConsole::SynchConsole(const char *in, const char *out)
{
  readAvail = new Semaphore("read avail", 0);
  writeDone = new Semaphore("write done", 0);
  getString = new Semaphore("get string", 0);
  putString = new Semaphore("put string", 0);
  console = new Console (in, out, ReadAvail, WriteDone, 0);
  getString->V();
  putString->V();  
}

void SynchConsole::putlock()
{
	putString->P ();
}

void SynchConsole::putunlock()
{
	putString->V ();
}

void SynchConsole::getlock()
{
	getString->P ();
}

void SynchConsole::getunlock()
{
	getString->V ();
}

SynchConsole::~SynchConsole()
{
  delete console;
  delete getString;
  delete writeDone;
  delete readAvail;
  delete putString;
}
void SynchConsole::SynchPutChar(int ch)
{
  console->PutChar (ch);
  writeDone->P ();
}

int SynchConsole::SynchGetChar()
{
  readAvail->P ();	// wait for character to arrive
  return console->GetChar ();
}

void SynchConsole::SynchPutString(const char s[])
{
  const char *ptr = s;
  while (*ptr != '\0')
  {
     console->PutChar (*ptr);
     writeDone->P ();
     ptr++;
  }
}

void SynchConsole::SynchPutInt(int n)
{
  static char s[50];
  snprintf(s,50,"%d",n);
  SynchPutString(s);
}

int SynchConsole::SynchGetInt()
{
  static char s[50];
  int n;
  SynchGetString(s,50);
  sscanf(s," %d ",&n);
  return n;
}



int copyStringFromMachine(int from, char *to, unsigned size)
{
	int c;
	char ch;
	int i;
	
	for (i = 0; i < (signed) size; i++)
	{
		machine->ReadMem(from+i,1,(int *)&ch);
		to[i] = ch;
		if (ch == '\0')
        {
			break;
        }
	}
	if (ch != '\0')
    	{
		to[i] = '\0';
        i++;
    	}
    
	return i;
}


int copyStringToMachine(int to, char *from, unsigned size)
{
	int c;
	char ch;
	unsigned i;
	
	for (i = 0; i < size; i++)
	{
	    machine->WriteMem(to+i,1,from[i]);
        if (from[i] == '\0')
            break;
    }
    if (from[i] != '\0')
    {
        ch = '\0';
		machine->WriteMem(to+i-1,1,ch);
    }
	return i;
}



void SynchConsole::SynchGetString(char *s, int n)
{
  int i = 0;
  int c;
  c = SynchGetChar();
  while ((i < n-1) && (c!=-1) && (c!='\n'))
  {
  	s[i] = c;
  	c = SynchGetChar();
  	i++;
  }
  if ((c == '\n')&&(i<n-1))
  	s[i++] = '\n';
  s[i] = '\0';
}
#endif // CHANGED


