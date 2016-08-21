library("ggplot2")
library("dplyr")
library("reshape2")

errors <- read.csv("errors.tsv", sep = "\t", stringsAsFactors = FALSE)

M <- dcast(errors, f + x ~ method, value.var = "err")

tmp <- errors %>%
    group_by(method) %>%
    summarize(
        mean_err = log10(mean(err)),
        max_err = log10(max(err))
    )

print(tmp)
