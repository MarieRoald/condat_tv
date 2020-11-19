# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import platform

if platform.system() == 'Windows':
	extra_args = ['/openmp', '/O3']
else:
	extra_args = ['-fopenmp', '-O3', '-ffast-math']

extensions = [
    Extension(
        "condat_tv.tv", ["src/condat_tv/tv.pyx"],
        extra_compile_args=extra_args,
        extra_link_args=extra_args,
    )
]

setup(
    ext_modules=cythonize(
            extensions,
        	compiler_directives={'embedsignature': True},
            language="c",
		),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
