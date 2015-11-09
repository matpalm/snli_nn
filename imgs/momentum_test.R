library(ggplot2)
library(reshape)
setwd("~/dev/snli_nn")

df = read.delim("out.tsv")
df$n = df$n_egs_trained
df$momentum = as.factor(df$momentum)
summary(df)

ggplot(df, aes(n, dev_acc)) + 
  geom_point(aes(color=momentum)) + 
  geom_smooth(aes(color=momentum)) + 
  facet_grid(~update_fn)
  
#ylim(0, 1)