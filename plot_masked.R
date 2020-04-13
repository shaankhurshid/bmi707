# Dependencies
library(data.table)
library(ggplot2)

# Script to plot masked acceleration data
actual <- fread('/Volumes/medpop_afib/skhurshid/bmi707/accel_flat.tsv')
transformed <- fread('/Volumes/medpop_afib/skhurshid/bmi707/transformed.tsv')

# Summarize
summary_actual <- apply(actual,MARGIN=1,FUN=mean,na.rm=T)
summary_transformed <- apply(transformed,MARGIN=1,FUN=mean,na.rm=T)

# Melt for ggplot2
x <- list(v1=b)
data <- melt(x)

# Plot distribution
ggplot() + geom_density(data=data,aes(x=value,fill=L1),color=NA,alpha=0.55) +
  scale_x_continuous(breaks=seq(5,60,5),expand=c(0,0),limits=c(5,60)) +
  scale_y_continuous(breaks=seq(0,0.08,0.01),expand=c(0,0),limits=c(0,0.08)) +
  scale_fill_manual(values=c("#2b8cbe"),name='',labels=c('Actual')) +
  theme(panel.background=element_blank(),axis.line=element_line(color='black'),legend.position=c(0.80,0.90),
        axis.text=element_text(size=20,color='black'),plot.margin=unit(c(0.5,0.5,0.5,0.5),'cm'),
        axis.title.y = element_text(size=20,margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(size=20),legend.text=element_text(size=20)) +
  labs(x='mean acceleration (milligravities)',y='density')
ggsave(filename='/Volumes/medpop_afib/skhurshid/bmi707/actual.pdf',height=2,width=3,
       scale=4,device='pdf')

# Plot distribution
x <- list(v1=a)
data <- melt(x)

ggplot() + geom_density(data=data,aes(x=value,fill=L1),color=NA,alpha=0.55) +
  scale_x_continuous(breaks=seq(5,60,5),expand=c(0,0),limits=c(5,60)) +
  scale_y_continuous(breaks=seq(0,0.08,0.01),expand=c(0,0),limits=c(0,0.08)) +
  scale_fill_manual(values=c("#f03b20"),name='',labels=c('Transformed')) +
  theme(panel.background=element_blank(),axis.line=element_line(color='black'),legend.position=c(0.80,0.90),
        axis.text=element_text(size=20,color='black'),plot.margin=unit(c(0.5,0.5,0.5,0.5),'cm'),
        axis.title.y = element_text(size=20,margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(size=20),legend.text=element_text(size=20)) +
  labs(x='mean acceleration (milligravities)',y='density')
ggsave(filename='/Volumes/medpop_afib/skhurshid/bmi707/transformed.pdf',height=2,width=3,
       scale=4,device='pdf')