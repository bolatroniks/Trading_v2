/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <python2.7/modsupport.h>
#include <math.h>

#include </usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h>

/* a preexisting C-level function you want to expose -- e.g: */
static double total(double* data, int len)
{
    double total = 0.0;
    int i;
    for(i=0; i<len; ++i)
        total += data[i];
    return total;
}

static PyObject* compute_labels (PyObject* self, PyObject* args)
{
    PyObject* seq;
    double *dbar;
    double result;
    int seqlen;
    int i;
    PyArrayObject *vecin, *vecout;

    if (!PyArg_ParseTuple(args, "O", &PyArray_Type, &vecin))
        return NULL;

    /*
    if (PyArg_ParseTuple(args, "O", &numpy_tmp_array)){
            /* Point our data to the data in the numpy pixel array */
    //        my_data_to_modify = (int*) numpy_tmp_array->data;
    //}

    seqlen = sizeof(vecin)/sizeof(vecin[0]);

    /*
    for(i=0; i < seqlen; i++) {
        PyObject *fitem;
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if(!item) {
            Py_DECREF(seq);
            free(dbar);
            return 0;
        }
        fitem = PyNumber_Float(item);
        if(!fitem) {
            Py_DECREF(seq);
            free(dbar);
            PyErr_SetString(PyExc_TypeError, "all items must be numbers");
            return 0;
        }
        dbar[i] = PyFloat_AS_DOUBLE(fitem);
        Py_DECREF(fitem);
    }*/

    /* clean up, compute, and return result */
    //Py_DECREF(seq);
    //result = total(dbar, seqlen);
    //free(dbar);
    //vecout = (PyArrayObject *)malloc();
    //return PyArray_Return(vecin);
    return Py_BuildValue("f", 0.0);;
}


/*  wrapped cosine function */
static PyObject* cos_func(PyObject* self, PyObject* args)
{
    double value;
    double answer;

    /*  parse the input, from python float to c double */
    if (!PyArg_ParseTuple(args, "d", &value))
        return NULL;
    /* if the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* call cos from libm */
    answer = cos(value);

    /*  construct the output from cos, from c double to python float */
    return Py_BuildValue("f", answer);
}

/*  define functions in module */
static PyMethodDef functions[] =
{
     {"compute_labels", compute_labels, METH_VARARGS, "compute labels"},
     {"cos_func", cos_func, METH_VARARGS, "evaluate the cosine"},
     {NULL, NULL, 0, NULL}
};

DL_EXPORT(void)
init_my_C_tools(void)
{
Py_InitModule("_my_C_tools", functions);
}
