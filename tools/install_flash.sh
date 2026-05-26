export CUDA_HOME=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

MAX_JOBS=8 pip install "flash-attn==2.8.3" --no-build-isolation
