library(ggplot2)
library(reshape)
setwd("~/dev/snli_nn")

df = read.delim("out.tsv")
df$n = df$n_egs_trained
summary(df)

ggplot(df, aes(n, dev_acc)) + 
  geom_point(aes(color=parse_mode)) + 
  geom_smooth(aes(color=parse_mode)) + 
  labs(title="token parse modes") #+ 
  #ylim(0, 1)