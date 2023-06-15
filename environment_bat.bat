@REM conda env config vars set PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1";%PATH% -n spineanalysis

@REM conda env config vars set LD_LIBRARY_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib";%LD_LIBRARY_PATH% -n spineanalysis

@REM conda env config vars set CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1" -n spineanalysis

@REM pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org
@REM pip install napari[all]


@REM pip install numpy
@REM pip install pyclesperanto_prototype
@REM pip install requests
@REM pip install csbdeep
@REM pip install pandas
@REM pip install albumentations
@REM pip install torchio
@REM pip install imgaug
@REM pip install torchsummary
@REM pip install pyyaml
@REM pip install pytest
@REM pip install colorcet
@REM pip install prettytable
@REM pip install numba
@REM pip install sklearn
@REM pip install FastGeodis --no-build-isolation
@REM pip install tensorboardX
cd spine-segment-pipeline
pip install -e . --no-build-isolation
cd ..