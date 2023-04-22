@REM conda env config vars set PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1";%PATH% -n spineanalysis

@REM conda env config vars set LD_LIBRARY_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib";%LD_LIBRARY_PATH% -n spineanalysis

@REM conda env config vars set CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1" -n spineanalysis

cd spine-segment-pipeline
pip install -e . --no-build-isolation
cd ..