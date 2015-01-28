setwd("/Volumes/MyBook4TB/DynamicNetworks/plots/average_weights_over_percentage/")
require(graphics)
t <- read.table("source_weights_for_r.txt", sep="\t")
t.T <- t(t)
t.T
headers <- t.T[1,]
headers
colnames(t.T) <- headers
t.T <- t.T[2:nrow(t.T),]
t.T

pdf("average_activity_increase.pdf")
par(mar=c(5.1, 4.1, 4.1, 10.1), xpd=F)
colors=rainbow(10)
plot(seq(0,100,1), as.numeric(t.T[,2]), log="y", type="b", xlab="Percentage of Nodes increased", 
     ylab="Average Activity per Node (log)", col=colors[1], pch=0, ylim=c(0.000000000001, 5), 
     lty=1)
title(expression(atop("Average" ~ "Activity" ~ "per" ~ "Node", ~ "after" ~ "%" ~ "increase" ~ "from" ~ a[0] ~ "to" ~ a[1])))
lines(seq(0,100,1), as.numeric(t.T[,3]), type="b", lty=1, col=colors[2], pch=1)
lines(seq(0,100,1), as.numeric(t.T[,4]), type="b", lty=2, col=colors[3], pch=2)
lines(seq(0,100,1), as.numeric(t.T[,5]), type="b", lty=2, col=colors[4], pch=3)
lines(seq(0,100,1), as.numeric(t.T[,6]), type="b", lty=3, col=colors[5], pch=4)
lines(seq(0,100,1), as.numeric(t.T[,7]), type="b", lty=3, col=colors[6], pch=10)
lines(seq(0,100,1), as.numeric(t.T[,8]), type="b", lty=4, col=colors[7], pch=6)
lines(seq(0,100,1), as.numeric(t.T[,9]), type="b", lty=4, col=colors[8], pch=7)
lines(seq(0,100,1), t.T[,10], type="b", lty=5, col=colors[9], pch=8)
lines(seq(0,100,1), t.T[,11], type="b", lty=5, col=colors[10], pch=9)
grid(col="gray", lwd=1.3)
legend("topright", c("Karate Random", "Karate Informed", "PA Random", "PA Informed", "R Random", "R Informed", "SBM Random", "SBM Informed", 
                     "DC SBM Random", "DC SBM Informed"), lty=c(1,1,2,2,3,3,4,4,5,5), pch=c(0,1,2,3,4,10,6,7,8,9), 
       col=colors, inset=c(-0.473,0), xpd=T, ncol=1)
dev.off()
