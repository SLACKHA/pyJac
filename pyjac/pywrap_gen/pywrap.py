from ..libgen import generate_library
import sys

def generate_setup(setupfile, home_dir, build_dir, libname):
	"""
	Helper method to fill in the template .in files
	"""
    with open(setupfile, 'r') as file:
        src = Template(file.read())
    file_data = {'homepath' : home_dir,
                 'buildpath' : build_dir,
                 'libname' : libname}
    src = src.safe_substitute(file_data)
    with open(setupfile[:setupfile.rindex('.in')], 'w') as file:
        file.write(src)

def generate_wrapper(lang, source_dir, out_dir=None, auto_diff=False):
	"""
	Generates a python wrapper for the given language and source files
	"""

	if lang != 'tchem':
		#first generate the library
		lib = generate_library(temp_lang, source_dir, shared=False, FD=auto_diff,
				extra_flags=None)

	setupfile = None
	if lang == 'c':
		setupfile = 'pyjacob_setup.py.in'
		if auto_diff:
			setupfile = 'adjacob_setup.py.in'
	elif lang == 'cuda':
		setupfile = 'pyjacob_cuda_setup.py.in'
	elif lang == 'tchem':
		setupfile = 'pytchem_setup.py.in'
	else:
		print('Language {} not recognized'.format(lang))
		sys.exit(-1)

	generate_setup(os.path.join(home_dir, setupfile), home_dir, build_dir,
		lib)
	subprocess.check_call(['python2.7', os.path.join(home_dir, setupfile), 
                       'build_ext', '--inplace'
                       ])

