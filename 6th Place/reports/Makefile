FIGURES = $(wildcard figures/*.png)

SLIDES_MD = $(wildcard *-slides.md)
SLIDES_PDF = $(patsubst %.md, %.pdf, $(SLIDES_MD))

all: $(SLIDES_PDF)

%-slides.pdf: %-slides.md $(FIGURES)
	pandoc --latex-engine=xelatex -t beamer \
		-V theme:metropolis \
		$< -o $@

gen_figures: gen_figures1

gen_figures1:
	../src/scripts/gen_figures.py ../data/processed/train_test.hdf5