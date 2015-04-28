args <- commandArgs(trailingOnly=T)
print(" ++ Setting work directory")
setwd(args[1])
print(" ++ Reading combined data")
data = read.table(args[2], sep="\t", header=T, colClasses="numeric")
print(" ++ Reading combined weights")
weights = read.table(args[3], sep="\t", header=T, colClasses="numeric")
graph_name = args[4]
scenario = args[5]
format = args[6]
cex_paper = 1.5
cex_size = 0.6
colors = c("#000000", "#858585")

print(" ++ Plotting activity")
if (format == "pdf") pdf(paste(graph_name, "_scenarios.pdf", sep="")) else png(paste(graph_name, "_scenarios.png", sep=""))
min_y = min(min(weights$sim_act), min(data$real_act_y))
max_y = max(max(weights$sim_act), max(data$real_act_y))
par(mar=c(5,5,4,5)+.1)
plot(weights$taus, weights$sim_act, type="l", xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type="l", lty=1, pch=1, col=colors[2])
title("Test", cex.main=cex_paper)
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col="#858585", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)
legend("topright", pch=c(NA,NA), col=colors, legend=c("Simulated Activity", "Observed Activity"), lty=c(1,1), cex=cex_paper)
dev.off()