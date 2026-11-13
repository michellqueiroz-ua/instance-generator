PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all figures clean

all: figures main.pdf

figures: make_figures.py
	$(PYTHON) make_figures.py

main.pdf: main.tex references.bib $(wildcard sections/*.tex) figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
