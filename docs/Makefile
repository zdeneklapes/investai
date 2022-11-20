# makefile pro preklad LaTeX verze Bc. prace
# makefile for compilation of the thesis
# (c) 2008 Michal Bidlo
# E-mail: bidlom AT fit vutbr cz
# Edited by: dytrych AT fit vutbr cz
#===========================================
# asi budete chtit prejmenovat / you will probably rename:
CO=xlapes02
CMD_PDFLATEX=pdflatex
CMD_BIBTEX=bibtex
FLAGS=-interaction=nonstopmode

all: $(LOGIN).pdf

pdf: $(LOGIN).pdf

$(LOGIN).ps: $(LOGIN).dvi
	dvips $(LOGIN)

$(LOGIN).pdf: clean
	$(CMD_PDFLATEX) $(FLAGS) $(LOGIN)
	-$(CMD_BIBTEX) $(LOGIN)
	$(CMD_PDFLATEX) $(FLAGS) $(LOGIN)
	$(CMD_PDFLATEX) $(FLAGS) $(LOGIN)
	$(MAKE) clean


$(LOGIN).dvi: $(LOGIN).tex $(LOGIN).bib
	latex $(LOGIN)
	-bibtex $(LOGIN)
	latex $(LOGIN)
	latex $(LOGIN)

clean:
	rm -f *.log *.dvi $(LOGIN).blg $(LOGIN).bbl $(LOGIN).toc *.aux $(LOGIN).out $(LOGIN).lof $(LOGIN).ptc
	find . -iname "*.log" -delete

clean_all: clean
	rm -f $(LOGIN).pdf

pack:
	tar czvf $(LOGIN).tar.gz *.tex *.bib *.bst ./template-fig/* ./bib-styles/* ./cls/* zadani.pdf $(LOGIN).pdf Makefile Changelog

rename:
	mv $(LOGIN).tex $(NAME).tex
	mv $(LOGIN)-01-kapitoly-chapters.tex $(NAME)-01-kapitoly-chapters.tex
	mv $(LOGIN)-01-kapitoly-chapters-en.tex $(NAME)-01-kapitoly-chapters-en.tex
	mv $(LOGIN)-20-literatura-bibliography.bib $(NAME)-20-literatura-bibliography.bib
	mv $(LOGIN)-30-prilohy-appendices.tex $(NAME)-30-prilohy-appendices.tex
	mv $(LOGIN)-30-prilohy-appendices-en.tex $(NAME)-30-prilohy-appendices-en.tex
	sed -i "s/$(LOGIN)-01-kapitoly-chapters/$(NAME)-01-kapitoly-chapters/g" $(NAME).tex
	sed -i "s/$(LOGIN)-01-kapitoly-chapters-en/$(NAME)-01-kapitoly-chapters-en/g" $(NAME).tex
	sed -i "s/$(LOGIN)-20-literatura-bibliography/$(NAME)-20-literatura-bibliography/g" $(NAME).tex
	sed -i "s/$(LOGIN)-30-prilohy-appendices/$(NAME)-30-prilohy-appendices/g" $(NAME).tex
	sed -i "s/$(LOGIN)-30-prilohy-appendices-en/$(NAME)-30-prilohy-appendices-en/g" $(NAME).tex
	sed -i "s/$(LOGIN)/$(NAME)/g" Makefile

# Pozor, vlna neresi vse (viz popis.txt) / Warning - vlna is not solving all problems (see description.txt)
vlna:
	vlna -l $(LOGIN)-*.tex

# Spocita normostrany / Count of standard pages
normostrany:
	echo "scale=2; `detex -n $(LOGIN)-[01]*.tex | sed s/"^ *"/""/ | sed s/"^	*"/""/ | wc -c`/1800;" | bc
