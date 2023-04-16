// exception.cc 
//      Entry point into the Nachos kernel from user programs.
//      There are two kinds of things that can cause control to
//      transfer back to here from user code:
//
//      syscall -- The user code explicitly requests to call a procedure
//      in the Nachos kernel.  Right now, the only function we support is
//      "Halt".
//
//      exceptions -- The user code does something that the CPU can't handle.
//      For instance, accessing memory that doesn't exist, arithmetic errors,
//      etc.  
//
//      Interrupts (which can also cause control to transfer from user
//      code into the Nachos kernel) are handled elsewhere.
//
// For now, this only handles the Halt() system call.
// Everything else core dumps.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "system.h"
#include "syscall.h"

//----------------------------------------------------------------------
// UpdatePC : Increments the Program Counter register in order to resume
// the user program immediately after the "syscall" instruction.
//----------------------------------------------------------------------
static void
UpdatePC ()
{
    int pc = machine->ReadRegister (PCReg);
    machine->WriteRegister (PrevPCReg, pc);
    pc = machine->ReadRegister (NextPCReg);
    machine->WriteRegister (PCReg, pc);
    pc += 4;
    machine->WriteRegister (NextPCReg, pc);
}


//----------------------------------------------------------------------
// ExceptionHandler
//      Entry point into the Nachos kernel.  Called when a user program
//      is executing, and either does a syscall, or generates an addressing
//      or arithmetic exception.
//
//      For system calls, the following is the calling convention:
//
//      system call code -- r2
//              arg1 -- r4
//              arg2 -- r5
//              arg3 -- r6
//              arg4 -- r7
//
//      The result of the system call, if any, must be put back into r2. 
//
// And don't forget to increment the pc before returning. (Or else you'll
// loop making the same system call forever!
//
//      "which" is the kind of exception.  The list of possible exceptions 
//      are in machine.h.
//----------------------------------------------------------------------

#include <iostream>
void
ExceptionHandler (ExceptionType which)
{
    int type = machine->ReadRegister (2);
    int c;
    if ((which == SyscallException) && (type == SC_Halt))
      {
	  DEBUG ('a', "Shutdown, initiated by user program.\n");
	  interrupt->Halt ();
      }
#ifdef CHANGED
    else if ((which == SyscallException) && (type == SC_PutChar))
      {synchconsole->putlock();
	  synchconsole->SynchPutChar(machine->ReadRegister(4));
	  synchconsole->putunlock();
      }
    else if ((which == SyscallException) && (type == SC_PutInt))
      {
      synchconsole->putlock();
	  synchconsole->SynchPutInt(machine->ReadRegister(4));
  	  synchconsole->putunlock();
      }      
    else if ((which == SyscallException) && (type == SC_GetChar))
      {
            synchconsole->getlock();
            c = synchconsole->SynchGetChar();
	    machine->WriteRegister(2,c);
	          synchconsole->getunlock();
      }
    else if ((which == SyscallException) && (type == SC_GetInt))
      {
            synchconsole->getlock();
            int a = synchconsole->SynchGetInt();
		    machine->WriteRegister(2,a);
			synchconsole->getunlock();
      }      
    else if ((which == SyscallException) && (type == SC_PutString))
      {
          int nb_ecrits;
          int i = 0;
          synchconsole->putlock();
          do
          {
             nb_ecrits = copyStringFromMachine(machine->ReadRegister(4)+i*(MAX_STRING_SIZE), string_buffer, MAX_STRING_SIZE);
     		 synchconsole->SynchPutString(string_buffer);
             i++;
          } while (nb_ecrits == MAX_STRING_SIZE+1);
          synchconsole->putunlock();
      }
    else if ((which == SyscallException) && (type == SC_GetString))
      {
			synchconsole->getlock();
        	int dest = machine->ReadRegister(4);
      		int taille_objectif = machine->ReadRegister(5);
      		char *s = new char[taille_objectif];
			synchconsole->SynchGetString(s, taille_objectif);
			copyStringToMachine(dest, s, taille_objectif);
			delete s;
			synchconsole->getunlock();	
      }
	else if ((which == SyscallException) && (type == SC_Exit))
	  {
	    interrupt->Exit (machine->ReadRegister(4));
	  }
#endif
    else
      {
	  printf ("Unexpected user mode exception %d %d\n", which, type);
	  ASSERT (FALSE);
      }

    // LB: Do not forget to increment the pc before returning!
    UpdatePC ();
    // End of addition
}
