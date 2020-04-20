#include "../include/binding.h"

static PyMethodDef C_moduleMethods[] = {
	{"regression_fit", regression_fit, METH_VARARGS, "Module C for regression"},
	{"regression_predict", regression_predict, METH_VARARGS, "Module C for regression"},
	{"neural_network_fit", neural_network_fit, METH_VARARGS, "Module C for neural network - fit"},
	{"neural_network_predict", neural_network_predict, METH_VARARGS, "Module C for neural network - predict"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef C_moduleModule = {
    PyModuleDef_HEAD_INIT,
    "C_module",
    "Python interface for the C_module library function",
    -1,
    C_moduleMethods
};

PyMODINIT_FUNC PyInit_C_module(void) {

    PyObject *module = PyModule_Create(&C_moduleModule);
	return module;
}
