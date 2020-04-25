from distutils.core import setup, Extension

def main():
    setup(name="C_module",
          description="Python interface for the C library function",
          author="GabrielChapo",
          ext_modules=[Extension("C_module", [
            "C_module/sources/binding.c",
            "C_module/sources/python_utils.c",
            "C_module/sources/error_functions.c",
            "C_module/sources/activation_functions.c",
            "C_module/sources/2D_matrix.c",
            "C_module/sources/neural_network.c",
            "C_module/sources/regression.c"
            ])])

if __name__ == "__main__":
    main()