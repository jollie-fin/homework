#ifdef CHANGED
#include "syscall.h"
/* Copyright (c) Microsoft Corporation. All rights reserved. */
/*  stdarg - defines ANSI-style macros for variable argument functions */

#ifndef _INC_STDARG
#define _INC_STDARG

#define __NEEDS_ROUND 1
#ifndef _VA_LIST_DEFINED
#define INT int
#if defined(arm)
#if defined(_M_ARM)
typedef char * va_list;
#elif defined(__HIGHC__) || defined(__GNUC__) || defined(ADS)
typedef void *va_list;
#else
typedef char *va_list[1];
#endif /*_M_ARM*/
#define _VA_LIST_DEFINED

#elif defined(__TCS__)
typedef void * _VA_LIST_;
typedef _VA_LIST_ va_list;
#define _VA_LIST_DEFINED

#elif defined(mips)
typedef char * va_list;
#ifdef __MIPSEB__
#define va_start(_ap_, _v_) \
  (_ap_ = (char*) __builtin_next_arg (_v_))
#define va_arg(_ap_, _type_) \
  ((_ap_ = (char *) ((__alignof__ (_type_) > 4 \
                       ? __ROUND((int)_ap_,8) : __ROUND((int)_ap_,4)) \
                     + __ROUND(sizeof(_type_),4))), \
   *(_type_ *) (void *) (_ap_ - __ROUND(sizeof(_type_),4)))
#define va_end(list)
#else
/* These are from NT */
#define va_start(ap,v) ap  = (va_list)&v + sizeof(v)
#define va_end(list)
#define va_arg(list, mode) ((mode *)(list =\
 (char *) ((((int)list + (__alignof__(mode)<=4?3:7)) &\
 (__alignof__(mode)<=4?-4:-8))+sizeof(mode))))[-1]
#endif
#define _VA_LIST_DEFINED

#elif defined(__H8300__)
/* later */

#else
typedef char *  va_list;
#define _VA_LIST_DEFINED
#endif
#endif /* _VA_LIST_DEFINED */


#ifdef  _M_IX86

#define _INTSIZEOF(n)   ((sizeof(n) + sizeof(INT) - 1) & ~(sizeof(INT) - 1)) 

#define va_start(ap,v)  (ap = (va_list)&v + _INTSIZEOF(v))
#define va_arg(ap,t)    (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
#define va_end(ap)      (ap = (va_list)0)

#elif   defined(_M_MRX000)

/* Use these types and definitions if generating code for MIPS */

#define va_start(ap,v)  ap = (va_list)&v + sizeof(v)
#define va_end(list)
#define va_arg(list, mode) ((mode *)(list = \
            (BYTE *) ((((INT)list + (__builtin_alignof(mode)<=4?3:7)) &\
            (__builtin_alignof(mode)<=4?-4:-8))+sizeof(mode))))[-1]

#elif   defined(arm)

#if defined(_M_ARM)

#define _INTSIZEOF(n)   ((sizeof(n) + sizeof(INT) - 1) & ~(sizeof(INT) - 1)) 

#define va_start(ap,v)  (ap = (va_list)&v + _INTSIZEOF(v))
#define va_arg(ap,t)    (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
#define va_end(ap)      (ap = (va_list)0)

#elif defined(__HIGHC__)

/*
 * __vararg_char is used as a pseudonym for "char". The compiler would
 * ordinarily warn about "misbehaviour" if an arbitrary pointer is cast 
 * to or from (char *) at high optimization levels.
 * To avoid this problem we use "__vararg_char" inplace of "char".
 * The compiler special-cases the type (__vararg_char *) and doesn't put
 * out the warning.
 */
typedef char __vararg_char;
#define va_start(_ap_,_parmN_) ((_ap_)=(__vararg_char *)&(_parmN_) + \
                ((sizeof(_parmN_)+(sizeof(int)-1)) & ~(sizeof(int)-1)))
#define _NNVAARG (-1)
#define va_arg(_ap_,_type_)\
      ( *(_type_ *) ((__vararg_char *)(\
                    _ap_ = (__vararg_char *)_ap_ + ((sizeof(_type_) +3) &~3)\
                              ) - ((sizeof(_type_) +3) &~3)\
                        + (sizeof(_type_)<=_NNVAARG?4-sizeof(_type_):0)\
                     )\
      )
#define va_end(_ap_) ((void)0)

#elif defined(__GNUC__) || defined(ADS)

/* Define __gnuc_va_list. */

#ifndef __GNUC_VA_LIST
#define __GNUC_VA_LIST
typedef void *__gnuc_va_list;
#endif

#define __va_rounded_size(TYPE)  \
  (((sizeof (TYPE) + sizeof (long) - 1) / sizeof (long)) * sizeof (long))

#ifdef ADS
#define va_start(AP, LASTARG) (void)((AP) = __va_start(LASTARG))
#else
#define va_start(AP,LASTARG) \
  (AP = ((__gnuc_va_list) __builtin_next_arg (LASTARG)))
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
/* This is for little-endian machines; small args are padded upward.  */
#define va_arg(AP, TYPE)                                                \
 (AP = (__gnuc_va_list) ((char *) (AP) + __va_rounded_size (TYPE)),     \
  *((TYPE *) (void *) ((char *) (AP) - __va_rounded_size (TYPE))))
#else
/* This is for big-endian machines; small args are padded downward.  */
#define va_arg(AP, TYPE)                                                \
 (AP = (__gnuc_va_list) ((char *) (AP) + __va_rounded_size (TYPE)),     \
  *((TYPE *) (void *) ((char *) (AP)                                    \
                       - ((sizeof (TYPE) < __va_rounded_size (char)     \
                           ? sizeof (TYPE) : __va_rounded_size (TYPE))))))
#endif

#define va_end(AP)      ((void) 0)

/* Copy __gnuc_va_list into another variable of this type.  */
#define __va_copy(dest, src) (dest) = (src)



/* compressed version of GCC's stdarg.h extra junk */

/* The macro _VA_LIST_DEFINED is used in Windows NT 3.5  */
#ifndef _VA_LIST_DEFINED
typedef __gnuc_va_list va_list;
#define _VA_LIST_DEFINED
#endif

#define _VA_LIST_   /* not sure about this */

#else

/* Use this for ARM and Norcroft C compiler */

#define __alignof(_t_) \
   ((char *)&(((struct{char __member1; \
                       ___type _t_ __member2;}*) 0)->__member2) - \
    (char *)0)
#define __alignuptotype(_p_,_t_) \
   ((char *)((int)(_p_) + (__alignof(_t_)-1) & ~(__alignof(_t_)-1)))

#define va_start(_ap_,_parmN_) \
    (void)(*(_ap_) = (char *)&(_parmN_) + sizeof(_parmN_))
#define va_end(_ap_) ((void)(*(_ap_) = (char *)-256))
#define va_arg(_ap_,_t_) \
   *(___type _t_ *)((*(_ap_)=__alignuptotype(*(_ap_),_t_)+sizeof(___type _t_))-\
                     sizeof(___type _t_))

#endif /* arm */

#elif defined(__TCS__)

/* Definitions for TriMedia compiler */
/* Rounding macros. */
#define _va_round_var(var)      ((sizeof(var)  < 4) ? 4 - sizeof(var)  : 0)
#define _va_roundup_var(var)    ((sizeof(var)  < 4) ? 4                : sizeof(var))
#define _va_round(type)         ((sizeof(type) < 4) ? 4 - sizeof(type) : 0)
#define _va_roundup(type)       ((sizeof(type) < 4) ? 4                : sizeof(type))

#define va_start(ap,lastarg) (ap = (va_list) (void *) &lastarg + _va_roundup_var(lastarg))

#define va_arg(ap,type) ((ap = ((char *)ap) + (_va_roundup(type))), *(type *) (ap + _va_round(type) - _va_roundup(type)))

#define va_end(ap)      ((void)0)


#elif defined(mips)

//#include <va-mips.h>


#elif defined(__H8300__)

#include <va-h8300.h>

#elif   defined(ppc) && defined(_MSC_VER)

#ifdef  __cplusplus
#define _ADDRESSOF(v)   ( &reinterpret_cast<const char &>(v) )
#else
#define _ADDRESSOF(v)   ( &(v) )
#endif

#define _VA_ALIGN(t)    8
#define _VA_IS_LEFT_JUSTIFIED(t) (sizeof(t) > _VA_ALIGN(t) || 0 != (sizeof(t) & (sizeof(t)-1)))

/* bytes that a type occupies in the argument list */
#define _INTSIZEOF(n)   ( (sizeof(n) + _VA_ALIGN(n) - 1) & ~(_VA_ALIGN(n) - 1) )
/* return 'ap' adjusted for type 't' in arglist */
#define _ALIGNIT(ap,t)  ( ((int)(ap) + _VA_ALIGN(t) - 1) & ~(_VA_ALIGN(t) - 1) )

#define va_start(ap,v)  ( ap = ( _VA_IS_LEFT_JUSTIFIED(v) ? (va_list)_ADDRESSOF(v) + _INTSIZEOF(v) \
                                                          : (va_list)(&(v)+1) ))

#define va_arg(ap,t) (ap = (va_list) (_ALIGNIT(ap, t) + _INTSIZEOF(t)), \
                      _VA_IS_LEFT_JUSTIFIED(t) ? *(t*)((ap) - _INTSIZEOF(t)) \
                                               : ((t*)(ap))[-1] )

#define va_end(ap)      ( ap = (va_list)0 )

#else

/* A guess at the proper definitions for other platforms */

#define _INTSIZEOF(n)   ((sizeof(n) + sizeof(INT) - 1) & ~(sizeof(INT) - 1))

#define va_start(ap,v)  (ap = (va_list)&v + _INTSIZEOF(v))
#define va_arg(ap,t)    (*(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)))
#define va_end(ap)      (ap = (va_list)0)

#endif

#endif  /* _INC_STDARG */


/*
 *  linux/lib/vsprintf.c
 *
 *  Copyright (C) 1991, 1992  Linus Torvalds
 */

/* vsprintf.c -- Lars Wirzenius & Linus Torvalds. */
/*
 * Wirzenius wrote this portably, Torvalds fucked it up :-)
 */

int isxdigit(char c)
{
	return ((c>='0' && c<='9') || (c>='A' && c<='F') || (c>='a' && c<='f'));
}

int isdigit(char c)
{
	return (c>='0' && c<='9');
}

int islower(char c)
{
	return (c>='a' && c<='z');
}

char toupper(char c)
{
	if (islower (c))
		return c + 'A' - 'a';
	else
		return c;
}

int strnlen(const char *s, int lmax)
{
	int i=0;
	while ((i < lmax) && (s[i] != '\0'))
		i++;
	return i;
}


unsigned long simple_strtoul(const char *cp,char **endp,unsigned int base)
{
	unsigned long result = 0,value;

	if (!base) {
		base = 10;
		if (*cp == '0') {
			base = 8;
			cp++;
			if ((*cp == 'x') && isxdigit(cp[1])) {
				cp++;
				base = 16;
			}
		}
	}
	while (isxdigit(*cp) && (value = isdigit(*cp) ? *cp-'0' : (islower(*cp)
	    ? toupper(*cp) : *cp)-'A'+10) < base) {
		result = result*base + value;
		cp++;
	}
	if (endp)
		*endp = (char *)cp;
	return result;
}

long simple_strtol(const char *cp,char **endp,unsigned int base)
{
	if(*cp=='-')
		return -simple_strtoul(cp+1,endp,base);
	return simple_strtoul(cp,endp,base);
}

/* we use this so that we can do without the ctype library */
#define is_digit(c)	((c) >= '0' && (c) <= '9')

static int skip_atoi(const char **s)
{
	int i=0;

	while (is_digit(**s))
		i = i*10 + *((*s)++) - '0';
	return i;
}

#define ZEROPAD	1		/* pad with zero */
#define SIGN	2		/* unsigned/signed long */
#define PLUS	4		/* show plus */
#define SPACE	8		/* space if plus */
#define LEFT	16		/* left justified */
#define SPECIAL	32		/* 0x */
#define LARGE	64		/* use 'ABCDEF' instead of 'abcdef' */

#define do_div(n,base) ({ \
int __res; \
__res = ((unsigned long) n) % (unsigned) base; \
n = ((unsigned long) n) / (unsigned) base; \
__res; })

static char _buffer[21] = {'\0'};
static int _indice;
static int _total;
#define tmpinit() do {_total = 0; _indice = 0;} while (0)
#define tmpputchar(c) do {_total++; _buffer[_indice++] = c; if (_indice == 20) {_indice = 0; _buffer[20] = '\0'; PutString(_buffer);}} while(0)
#define tmpflush() do {_buffer[_indice] = '\0'; PutString(_buffer); _indice = 0;} while (0)
#define tmpnbecrits() _total
static void number(long num, int base, int size, int precision
	,int type)
{
	char c,sign,tmp[66];
	const char *digits="0123456789abcdefghijklmnopqrstuvwxyz";
	int i;

	if (type & LARGE)
		digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	if (type & LEFT)
		type &= ~ZEROPAD;
	if (base < 2 || base > 36)
		return;
	c = (type & ZEROPAD) ? '0' : ' ';
	sign = 0;
	if (type & SIGN) {
		if (num < 0) {
			sign = '-';
			num = -num;
			size--;
		} else if (type & PLUS) {
			sign = '+';
			size--;
		} else if (type & SPACE) {
			sign = ' ';
			size--;
		}
	}
	if (type & SPECIAL) {
		if (base == 16)
			size -= 2;
		else if (base == 8)
			size--;
	}
	i = 0;
	if (num == 0)
		tmp[i++]='0';
	else while (num != 0)
		tmp[i++] = digits[do_div(num,base)];
	if (i > precision)
		precision = i;
	size -= precision;
	if (!(type&(ZEROPAD+LEFT)))
		while(size-->0)
			tmpputchar(' ');
	if (sign)
		tmpputchar(sign);
	if (type & SPECIAL) {
		if (base==8)
			tmpputchar('0');
		else if (base==16) {
			tmpputchar('0');
			tmpputchar(digits[33]);
		}
	}
	if (!(type & LEFT))
		while (size-- > 0)
			tmpputchar(c);
	while (i < precision--)
		tmpputchar('0');
	while (i-- > 0)
		tmpputchar(tmp[i]);
	while (size-- > 0)
		tmpputchar(' ');
}

int vprintf(const char *fmt, va_list args)
{
	int len;
	unsigned long num;
	int i, base;
	const char *s;

	int flags;		/* flags to number() */

	int field_width;	/* width of output field */
	int precision;		/* min. # of digits for integers; max
				   number of chars for from string */
	int qualifier;		/* 'h', 'l', or 'L' for integer fields */

    tmpinit();

	for (; *fmt ; ++fmt) {
		if (*fmt != '%') {
			tmpputchar(*fmt);
			continue;
		}
			
		/* process flags */
		flags = 0;
		repeat:
			++fmt;		/* this also skips first '%' */
			switch (*fmt) {
				case '-': flags |= LEFT; goto repeat;
				case '+': flags |= PLUS; goto repeat;
				case ' ': flags |= SPACE; goto repeat;
				case '#': flags |= SPECIAL; goto repeat;
				case '0': flags |= ZEROPAD; goto repeat;
				}
		
		/* get field width */
		field_width = -1;
		if (is_digit(*fmt))
			field_width = skip_atoi(&fmt);
		else if (*fmt == '*') {
			++fmt;
			/* it's the next argument */
			field_width = va_arg(args, int);
			if (field_width < 0) {
				field_width = -field_width;
				flags |= LEFT;
			}
		}

		/* get the precision */
		precision = -1;
		if (*fmt == '.') {
			++fmt;	
			if (is_digit(*fmt))
				precision = skip_atoi(&fmt);
			else if (*fmt == '*') {
				++fmt;
				/* it's the next argument */
				precision = va_arg(args, int);
			}
			if (precision < 0)
				precision = 0;
		}

		/* get the conversion qualifier */
		qualifier = -1;
		if (*fmt == 'h' || *fmt == 'l' || *fmt == 'L') {
			qualifier = *fmt;
			++fmt;
		}

		/* default base */
		base = 10;

		switch (*fmt) {
		case 'c':
			if (!(flags & LEFT))
				while (--field_width > 0)
					tmpputchar(' ');
			tmpputchar((unsigned char) va_arg(args, int));
			while (--field_width > 0)
				tmpputchar(' ');
			continue;

		case 's':
			s = va_arg(args, char *);
			if (!s)
				s = "<NULL>";

			len = strnlen(s, precision);

			if (!(flags & LEFT))
				while (len < field_width--)
					tmpputchar(' ');
			for (i = 0; i < len; ++i)
				tmpputchar(*s++);
			while (len < field_width--)
				tmpputchar(' ');
			continue;

		case 'p':
			if (field_width == -1) {
				field_width = 2*sizeof(void *);
				flags |= ZEROPAD;
			}
			number(
				(unsigned long) va_arg(args, void *), 16,
				field_width, precision, flags);
			continue;


		case 'n':
			if (qualifier == 'l') {
				long * ip = va_arg(args, long *);
				*ip = tmpnbecrits();
			} else {
				int * ip = va_arg(args, int *);
				*ip = tmpnbecrits();
			}
			continue;

		/* integer number formats - set up the flags and "break" */
		case 'o':
			base = 8;
			break;

		case 'X':
			flags |= LARGE;
		case 'x':
			base = 16;
			break;

		case 'd':
		case 'i':
			flags |= SIGN;
		case 'u':
			break;

		default:
			if (*fmt != '%')
				tmpputchar('%');
			if (*fmt)
				tmpputchar(*fmt);
			else
				--fmt;
			continue;
		}
		if (qualifier == 'l')
			num = va_arg(args, unsigned long);
		else if (qualifier == 'h') {
			num = (unsigned short) va_arg(args, int);
			if (flags & SIGN)
				num = (short) num;
		} else if (flags & SIGN)
			num = va_arg(args, int);
		else
			num = va_arg(args, unsigned int);
		number(num, base, field_width, precision, flags);
	}
	tmpflush ();

	return tmpnbecrits();
}

int printf(const char *fmt, ...)
{
	va_list args;
	int i;

	va_start(args, fmt);
	i=vprintf(fmt,args);
	va_end(args);
	return i;
	
}

#endif
