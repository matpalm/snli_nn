library(ggplot2)
library(reshape)
setwd("~/dev/snli_nn")

df = read.delim("out.tsv")
df$n = df$n_egs_trained
df$keep_prob = as.factor(df$keep_prob)
summary(df)

ggplot(df, aes(n, dev_acc)) + 
  geom_point(aes(color=keep_prob)) + 
  geom_smooth(aes(color=keep_prob)) + 
  labs(title="dropout") + ylim(0, 1)