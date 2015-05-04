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
cex_size = 1
colors = c("#000000", "#858585", "red", "green", "blue", "pink", "cyan", "darkorange", "brown")
legend_text <- append(list("Simulated Activity", "Observed Activity"), step_values)
file_name = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), ".pdf", sep="")

clean_ratios <- data$ratios[!is.na(data$ratios)]
clean_weights <- weights[!is.na(weights)]
marker = floor((length(clean_ratios) / 3) * 2)

pch_skip = as.numeric(args[9])

print(paste("Marker: ", marker, sep=""))

print(" ++ Plotting activity")
if (format == "pdf") pdf(file_name) else png(file_name)
min_y = min(min(clean_weights), min(data$real_act_y))
max_y = max(max(clean_weights), max(data$real_act_y))
par(mar=c(8,5,4,5)+.1, xpd=T)
plot(weights$taus, weights$sim_act, type="o", pch=c(1, rep(NA, pch_skip)), xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
points(tail(weights$taus, n=1), tail(weights$sim_act, n=1), pch=1, col=colors[1], cex=cex_size)
lines(data$real_act_x, data$real_act_y, type="o", pch=2, lty=1, col=colors[2], cex=cex_size)
i = 3
while(i <= length(weights)) {
  lines(weights$taus, weights[,i], type="o", pch=c(i, rep(NA, pch_skip)), lty=1, col=colors[i], cex=cex_size)
  points(tail(weights$taus, n=1), tail(weights[,i], n=1), pch=i, col=colors[i], cex=cex_size)
  i = i +1
}
legend("bottom", inset=c(0, -0.3), pch=seq(1, length(weights), 1), col=colors, legend=legend_text, lty=1, cex=0.7, ncol=2)

par(mar=c(8,5,4,5)+.1, xpd=F)
title(substitute(atop(scenario * " for " * bold(graph_name), 
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

pdf("test.pdf", width="10", height="1")
plot.new()
par(xpd=T)
legend("center", inset=c(0, -0.3), pch=seq(1, length(weights), 1), col=colors, legend=legend_text, lty=1, cex=0.3, horiz=TRUE)
par(xpd=F)
dev.off