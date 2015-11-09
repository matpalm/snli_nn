library(ggplot2)
library(reshape)
setwd("~/dev/snli_nn")

df = read.delim("out.tsv")
df$n = df$n_egs_trained
df$learning_rate = as.factor(df$learning_rate)
summary(df)

ggplot(df, aes(n, train_cost)) + 
  geom_point(aes(color=update_fn)) + 
  geom_smooth(aes(color=update_fn)) + 
  labs(title="seq2seq lr") + facet_grid(~learning_rate)
#ylim(0, 1)