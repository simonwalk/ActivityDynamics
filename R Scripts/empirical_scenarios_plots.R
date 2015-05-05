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
cex_paper = 2
cex_size = 2
colors = c("#000000", "#858585", "red", "green", "blue", "purple", "cyan", "darkorange", "brown")
line_colors = c("red", "green")
legend_text <- append(list("Simulated Activity", "Observed Activity"), step_values)
file_name = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), ".pdf", sep="")

clean_ratios <- data$ratios[!is.na(data$ratios)]
clean_weights <- weights[!is.na(weights)]
marker = floor((length(clean_ratios) / 3) * 2)

pch_skip = as.numeric(args[9])

x_values <- seq(0,length(data$real_act_x)-1,1)

print(x_values)

print(paste("Marker: ", marker, sep=""))

print(" ++ Plotting activity")
if (format == "pdf") pdf(file_name) else png(file_name)
min_y = min(min(clean_weights), min(data$real_act_y))
max_y = max(max(clean_weights), max(data$real_act_y))
par(mar=c(5,5,1,1)+.1)
plot(x_values, weights$sim_act, type="o", pch=c(1), xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
#points(tail(weights$taus, n=1), tail(weights$sim_act, n=1), pch=1, col=colors[1], cex=cex_size)
lines(data$real_act_x, data$real_act_y, type="o", pch=2, lty=1, col=colors[2], cex=cex_size)
i = 2
while(i <= length(weights)) {
  print(names(weights)[i])
  lines(x_values, weights[,i], type="o", pch=c(i+1), lty=1, col=colors[i+1], cex=cex_size)
  #points(tail(weights$taus, n=1), tail(weights[,i], n=1), pch=i, col=colors[i], cex=cex_size)
  i = i +1
}
#legend("bottom", inset=c(0, -0.3), pch=seq(1, length(weights)+1, 1), col=colors, legend=legend_text, lty=1, cex=0.7, ncol=2)


#title(substitute(atop(scenario * " for " * bold(graph_name), 
#                      "with " * rand_itas * " Random Iterations"),
#                 list(scenario = scenario, graph_name = graph_name, rand_itas = rand_itas)), cex.main=cex_paper)
abline(v=marker, lty=2, col=colors[1])
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col="#858585", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)

dev.off()

pdf(paste(scenario, "_legend.pdf"), width=10, height=1)
par(mar=c(0,0,0,0))
plot.new()
legend("center", pch=seq(1, length(weights)+1, 1), col=colors, legend=legend_text, lty=1, cex=1.2, ncol=4)
dev.off