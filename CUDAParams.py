"""Module containing parameters that control CUDA code generation
"""

class CudaMemStrats:
	Global, Local = range(2)

MemoryStrategy = CudaMemStrats.Global

def is_global():
	return MemoryStrategy == CudaMemStrats.Global