args <- commandArgs(trailingOnly=T)
print(" ++ Setting work directory")
setwd(args[1])
print(" ++ Reading combined data")
graph_name = args[2]
data <- read.table(args[3], colClasses = "numeric")

library(ggplot2)

pdf(paste(graph_name, "_deg_dis.pdf", sep=""))#, width=14, height=5)

breaks <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 500, 5000)

data2 <- transform(data, groupdata = cut(data$V1, breaks=breaks, right=T, include.lowest=T))

qplot(x=groupdata, data=data2, stat="bin", xlab="Degree", ylab="# Users") + scale_x_discrete(labels=c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11-25", "26-50", "51-100", "101-500", "501-5000")) +
theme(axis.text.x = element_text(angle = 45, hjust = 1, size=20, color="black"), panel.background = element_blank(),
axis.title=element_text(size=20), axis.text.y = element_text(size=20, color="black")) +
stat_bin(geom="text", aes(label=..count.., vjust=-0.5))

dev.off()