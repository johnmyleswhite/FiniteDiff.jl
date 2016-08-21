check_accuracy:
	julia benchmarks/accuracy/errors.jl
	Rscript benchmarks/accuracy/errors.R
	rm errors.tsv

all: check_accuracy
