from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vulnerability_benchmark_ops',
    ext_modules=[
        CUDAExtension(
            name='vulnerability_benchmark_ops',
            sources=[
                'modules/custom_operator.cpp',
                'modules/kernel.cu'
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
