#CParams.py

"""
Various parameters that control C Jacobian unrolling

Parameters
----------

Jacob_Unroll : int
	Limit for number of reactions in each Jacobian reaction update subfile
Jacob_Spec_Unroll : int
	The number of species to place in each Jacobian species update subfile
Max_Lines : int
	Limit for number of lines of each Jacobian reaction update subfile
Max_Spec_Lines : int
	Limit for number of lines of each Jacobian species update subfile
"""

Jacob_Unroll = 40
Jacob_Spec_Unroll = 40
Max_Lines = 50000
Max_Spec_Lines = 50000
