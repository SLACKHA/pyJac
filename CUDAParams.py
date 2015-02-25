"""Module containing parameters that control CUDA code generation
"""

class CudaMemStrats:
	Global, Local = range(2)

MemoryStrategy = CudaMemStrats.Local

def is_global():
	return MemoryStrategy == CudaMemStrats.Global

l1_size = 48000 / 8 #doubles
desired_thread_count = 64 #threads / block