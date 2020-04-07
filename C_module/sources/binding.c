#include "../include/binding.h"

/*
	In order to call the methods defined in your module,
	you’ll need to tell the Python interpreter about them first.
	To do this, you can use PyMethodDef.
	This is a structure with 4 members representing a single method
	in your module.
	Each individual member of the struct holds the following info:
		$ "fputs" is the name the user would write to invoke this
		particular function.
		
		$ method_fputs is the name of the C function to invoke.
		
		$ METH_VARARGS is a flag that tells the interpreter
		that the function will accept two arguments of type PyObject*:
			$ self is the module object.
			$ args is a tuple containing the actual arguments to your
			function. As explained previously, these arguments are
			unpacked using PyArg_ParseTuple().
		
		$ The final string is a value to represent the method docstring.
*/
static PyMethodDef C_moduleMethods[] = {
	{"linear_regression_fit", linear_regression_fit, METH_VARARGS, "Module C for linear regression - fit"},
	{"linear_regression_predict", linear_regression_predict, METH_VARARGS, "Module C for linear regression - predict"},
	{"logistic_regression_fit", logistic_regression_fit, METH_VARARGS, "Module C for logistic regression - fit"},
	{"logistic_regression_predict", logistic_regression_predict, METH_VARARGS, "Module C for logistic regression - predict"},
	{NULL, NULL, 0, NULL}
};

/*
	The PyModuleDef struct holds information about your module itself.
	There are a total of 9 members in this struct, but not all of them are required.
	In the code block above, you initialize the following five:

		$ PyModuleDef_HEAD_INIT is a member of type PyModuleDef_Base,
		which is advised to have just this one value.

		$ "fputs" is the name of your Python C extension module.

		$ The string is the value that represents your module docstring.
		You can use NULL to have no docstring, or you can specify a docstring
		by passing a const char * as shown in the snippet above.
		It is of type Py_ssize_t. You can also use PyDoc_STRVAR()
		to define a docstring for your module.

		$ -1 is the amount of memory needed to store your program state.
		It’s helpful when your module is used in multiple sub-interpreters,
		and it can have the following values:
			$ A negative value indicates that this module doesn’t have support for sub-interpreters.
			$ A non-negative value enables the re-initialization of your module. It also specifies the memory requirement of your module to be allocated on each sub-interpreter session.
		
		$ FputsMethods is the reference to your method table.
		This is the array of PyMethodDef structs you defined earlier.
*/
static struct PyModuleDef C_moduleModule = {
    PyModuleDef_HEAD_INIT,
    "C_module",
    "Python interface for the C_module library function",
    -1,
    C_moduleMethods
};

/*
	When a Python program imports your module for the first time,
	it will call PyInit_fputs().

	Add int constant by name:
    >PyModule_AddIntConstant(module, "FPUTS_FLAG", 64);

	Initialize new exception object and Add exception object to your module:
    >StringTooShortError = PyErr_NewException("fputs.StringTooShortError", NULL, NULL);
    >PyModule_AddObject(module, "StringTooShortError", StringTooShortError);

*/
PyMODINIT_FUNC PyInit_C_module(void) {
    // Assign module value
    PyObject *module = PyModule_Create(&C_moduleModule);

	return module;
}
