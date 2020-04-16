#include "../include/binding.h"

static PyMethodDef C_moduleMethods[] = {
	{"linear_regression_fit", linear_regression_fit, METH_VARARGS, "Module C for linear regression - fit"},
	{"linear_regression_predict", linear_regression_predict, METH_VARARGS, "Module C for linear regression - predict"},
	{"logistic_regression_fit", logistic_regression_fit, METH_VARARGS, "Module C for logistic regression - fit"},
	{"logistic_regression_predict", logistic_regression_predict, METH_VARARGS, "Module C for logistic regression - predict"},
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
