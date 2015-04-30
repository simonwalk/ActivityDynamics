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
rand_itas = args[7]
step_values = as.vector(unlist(strsplit(args[8], ", ")), mode="list")
cex_paper = 1.5
cex_size = 0.6
colors = c("#000000", "#858585", "red", "green", "blue", "pink", "cyan", "darkorange", "brown")
legend_text <- append(list("Simulated Activity", "Observed Activity"), step_values)
file_name = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), ".pdf", sep="")

clean_ratios <- data$ratios[!is.na(data$ratios)]
marker = floor((length(clean_ratios) / 3) * 2)

print(paste("Marker: ", marker, sep=""))

print(" ++ Plotting activity")
if (format == "pdf") pdf(file_name) else png(file_name)
min_y = min(min(weights$sim_act), min(data$real_act_y))
max_y = max(max(weights$sim_act), max(data$real_act_y))
par(mar=c(8,5,4,5)+.1, xpd=T)
plot(weights$taus, weights$sim_act, type="l", xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type="l", lty=1, pch=1, col=colors[2])

i = 3
while(i <= length(weights)) {
  lines(weights$taus, weights[,i], type="l", lty=1, col=colors[i])
  i = i +1
}
legend("bottom", inset=c(0, -0.3), pch=c(NA,NA), col=colors, legend=legend_text, lty=c(1,1), cex=0.7, horiz=T)

par(mar=c(8,5,4,5)+.1, xpd=F)
title(substitute(atop(scenario * " for " * graph_name, 
                      "with " * rand_itas * " Random Iterations"),
                 list(scenario = scenario, graph_name = graph_name, rand_itas = rand_itas)), cex.main=cex_paper)
abline(v=marker, lty=2, col=colors[1])
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col="#858585", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)

dev.off()