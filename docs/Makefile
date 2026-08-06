# Minimal makefile for Sphinx documentation

# Set from the command line or the environment, e.g. `make html SPHINXOPTS=`
# to build without treating warnings as errors. The default matches CI, so a
# local build that passes is one that will pass there too.
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Route all unknown targets to Sphinx's "make mode".
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
