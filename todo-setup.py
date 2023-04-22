#!/usr/bin/env python
import os
import setuptools

import numpy as np

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

# NOTE: If skeletontricks.cpp does not exist, you must run
# cython -3 --cplus ./ext/skeletontricks/skeletontricks.pyx

setuptools.setup(
  name="mfspine",
  version="1.0.0",
  author="Fjscah",
  setup_requires=["numpy"],
  install_requires=[
    "sys",
"QtPy==1.9.0",
"matplotlib==3.2.2",

"pytest==5.4.3",
"scipy==1.5.0",
"plotly==4.13.0",
"h5py==2.10.0",
"scikit-learn"
"numpy==1.23.4",
"tqdm==4.47.0",
"imgaug==0.3.0",
"pandas==1.0.5",
"sympy==1.6.1",
"keras==2.8.0",
"SQLAlchemy==1.3.18",
"pip==20.1.1",
"batchgenerators==0.24",
"btrack==0.4.6",
"csbdeep==0.7.2",
"magicgui==0.6.1",
"napari==0.4.17",
"nets==0.0.3.1",
"nibabel==4.0.2",
"paddle==1.0.2",
"Pillow==9.3.0",
"pint==0.20.1",
"psygnal==0.6.1",
"PyYAML==6.0",
"scikit_learn==1.1.3",
"skimage==0.0",

"xgboost==1.7.1",
  ],
  extras_require={
    'tif': [ 'tifffile' ],
  },
  python_requires=">=3.6.0,<4.0.0",

  
  packages=setuptools.find_packages(),
  description="Skeletonize densely labeled image volumes.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  keywords = "volumetric-data numpy teasar skeletonization centerline medial-axis-transform centerline-extraction computer-vision-alogithms connectomics image-processing biomedical-image-processing voxel",
  url = "https://github.com/mfspine/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],
  entry_points={
    "console_scripts": [
      "mfspine=mfspine:main"
    ],
  },
)

