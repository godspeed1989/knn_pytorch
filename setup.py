import os, glob, shutil
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    a = setup(
        name='gather_nn_cuda',
        ext_modules=[
            CUDAExtension(
                name='gather_nn_cuda',
                sources=['gather_nn.cpp', 'gather_nn_kernel.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']},
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    files = glob.iglob(os.path.join('./build/lib*/', "*.so"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, './')
