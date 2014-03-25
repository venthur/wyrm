DOCDIR=doc
APIDOCDIR=$(DOCDIR)/api
SRCDIR=wyrm

GH_PAGES=gh_pages


.PHONY: test pyflakes doc

all: test


test: pyflakes
	nosetests --with-coverage --cover-package=wyrm


pyflakes:
	- PYFLAKES_NODOCTEST=1 pyflakes wyrm
	- PYFLAKES_NODOCTEST=1 pyflakes test
	- PYFLAKES_NODOCTEST=1 pyflakes examples


doc:
	rm -rf $(APIDOCDIR)
	mkdir -p $(APIDOCDIR)
	sphinx-apidoc -o $(APIDOCDIR) $(SRCDIR)
	make -C $(DOCDIR) html


ghpages: doc
	rm -rf $(GH_PAGES)
	git clone git@github.com:venthur/wyrm $(GH_PAGES)
	cd $(GH_PAGES); git checkout gh-pages
	rm -rf $(GH_PAGES)/*
	cp -r $(DOCDIR)/_build/html/* $(GH_PAGES)
	touch $(GH_PAGES)/.nojekyll
	@echo
	@echo cd into $(GH_PAGES), review, commit and pull
	@echo


pypi:
	python setup.py stdist upload


clean:
	rm -rf $(APIDOCDIR)
	rm -rf $(GH_PAGES)
	make -C $(DOCDIR) clean
	rm -rf .coverage
	rm -rf build
	rm -rf dist
	rm -rf MANIFEST
