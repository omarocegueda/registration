# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:37:17 2013

@author: khayyam
"""
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#ext_modules = [Extension("tensorFieldUtils", ["tensorFieldUtils.pyx"])]
ext_modules = [Extension("tensorFieldUtils", ["tensorFieldUtilsPYX.pyx", "tensorFieldUtilsCPP.cpp"],include_dirs=get_numpy_include_dirs(), language="c++")]
ext_modules.append(Extension("ecqmmf", ["ecqmmfCYTHON.pyx", "ecqmmfCPP.cpp","bitsCPP.cpp", "ecqmmf_regCPP.cpp"],include_dirs=get_numpy_include_dirs(), language="c++"))
setup(
  name = 'Tensor Field Utilities',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)


#from numpy.distutils.misc_util import get_numpy_include_dirs
#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext
#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("ornlm_module", ["ornlm_module.pyx", "ornlm.cpp", "upfirdn.cpp"],include_dirs=get_numpy_include_dirs(), language="c++")]
#)
