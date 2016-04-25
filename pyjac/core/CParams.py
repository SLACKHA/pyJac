#CParams.py

"""
Various parameters that control C Jacobian unrolling

Parameters
----------

Jacob_Unroll : int
	The number of reactions to attempt to place in each Jacobian reaction update subfile
Jacob_Spec_Unroll : int
	The number of species to attempt to place in each Jacobian species update subfile
Max_Lines : int
	The number of lines to attempt to limit each Jacobian reaction update subfile to
Max_Spec_Lines : int
	The number of lines to attempt to limit each Jacobian species update subfile to
"""

Jacob_Unroll = 40
Jacob_Spec_Unroll = 40
Max_Lines = 50000
Max_Spec_Lines = 50000
