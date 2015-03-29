"""Module containing parameters that control CUDA code generation
"""

class CudaMemStrats:
	Global, Local = range(2)

MemoryStrategy = CudaMemStrats.Local

def is_global():
	return MemoryStrategy == CudaMemStrats.Global

def get_L1_size(L1_Preferred):
	if L1_Preferred:
		return 49152 / 8 #doubles
	else:
		return 16384 / 8 #doubles

def get_shared_size(L1_Preferred):
	if not L1_Preferred:
		return 49152 / 8 #doubles
	else:
		return 16384 / 8 #doubles

def get_register_count(num_blocks, num_threads):
	return max(min((32768 / num_blocks) / num_threads, 63), 1)