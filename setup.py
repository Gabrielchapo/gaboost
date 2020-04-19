from distutils.core import setup, Extension

"""
    The code block above shows the standard arguments that
    are passed to setup(). Take a closer look at the last
    positional argument, ext_modules. This takes a list of
    objects of the Extensions class. An object of the Extensions
    class describes a single C or C++ extension module in a setup
    script. Here, you pass two keyword arguments to its constructor,
    namely:
        
        $ name: is the name of the module.
        $ ext_modules: [filename] is a list of paths to files with the source code,
        relative to the setup script.
"""

def main():
    setup(name="C_module",
          version="1.0.0",
          description="Python interface for the C library function",
          author="GabrielChapo",
          author_email="gabrieldrai@yahoo.fr",
          ext_modules=[Extension("C_module", [
            "C_module/sources/binding.c",
            "C_module/sources/python_utils.c",
            "C_module/sources/error_functions.c",
            "C_module/sources/2D_matrix.c",
            "C_module/sources/linear_regression.c",
            "C_module/sources/logistic_regression.c",
            "C_module/sources/neural_network.c"
            ])])

if __name__ == "__main__":
    main()