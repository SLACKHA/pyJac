"""Module containing parameters that control CUDA code generation
"""

class CudaMemStrats:
	Global, Local = range(2)

MemoryStrategy = CudaMemStrats.Local

def is_global():
	return MemoryStrategy == CudaMemStrats.Global