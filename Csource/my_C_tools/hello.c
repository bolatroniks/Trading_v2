#include <Python.h>

PyObject*
hello(PyObject* self)
{
printf("Hello world!\n");
Py_RETURN_NONE;
}

static PyMethodDef functions[] = {
{"hello", (PyCFunction)hello, METH_NOARGS},
{NULL, NULL, 0, NULL},
};

DL_EXPORT(void)
init_hello(void)
{
Py_InitModule("_hello", functions);
}
