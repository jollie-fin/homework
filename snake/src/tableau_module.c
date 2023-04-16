#include <Python.h>
#include "Imaging.h"

typedef struct {
    PyObject_HEAD
    Imaging image;
    ImagingAccess access;
} ImagingObject;

static inline int
_getrgb(PyObject* rgb, int* r, int *g, int *b)
{
    PyObject* value;

    if (!PyTuple_Check(rgb) || PyTuple_GET_SIZE(xy) != 3)
        goto badarg;
        
    value = PyTuple_GET_ITEM(rgb, 0);
    if (PyInt_Check(value))
        *r = PyInt_AS_LONG(value);
    else
        goto badval;

    value = PyTuple_GET_ITEM(rgb, 1);
    if (PyInt_Check(value))
        *g = PyInt_AS_LONG(value);
    else
        goto badval;

    value = PyTuple_GET_ITEM(rgb, 2);
    if (PyInt_Check(value))
        *b = PyInt_AS_LONG(value);
    else
        goto badval;

    return 0;

  badarg:
    PyErr_SetString(
        PyExc_TypeError,
        "argument must be sequence of length 2"
        );
    return -1;

  badval:
    PyErr_SetString(
        PyExc_TypeError,
        "an integer is required"
        );
    return -1;
}

static PyObject* 
rempli_couleur(ImagingObject* self, PyObject* args)
{
    PyObject* rgb;
    int r, g, b;

    if (PyTuple_GET_SIZE(args) != 1) {
        PyErr_SetString(
            PyExc_TypeError,
            "argument 1 must be sequence of length 3"
            );
        return NULL;
    }


    rgb = PyTuple_GET_ITEM(args, 0);

    if (_getrgb(rgb, &r, &g, &b))
        return NULL;

    if (self->access == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }

	return Py_None;
//    return getpixel(self->image, self->access, x, y);
}

static PyMethodDef Methods[] = {
    ...
    {"rempli_couleur",  rempli_couleur, METH_VARARGS,
     "Rempli couleur"},
    ...
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
inittableau(void)
{
    (void) Py_InitModule("Tableau", Methods);
}

int
main(int argc, char *argv[])
{
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    inittableau();
}

