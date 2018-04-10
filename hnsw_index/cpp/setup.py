from distutils.core import setup, Extension
from Cython.Build import cythonize


ext = Extension(
    "pyhnsw",
    sources=["pyhnsw.pyx", "hnsw.cpp", "dumps.cpp", "utils.cpp"],
    language="c++",
)

setup(
    name="pyhnsw",
    ext_modules=cythonize(ext, language='c++'),
)
