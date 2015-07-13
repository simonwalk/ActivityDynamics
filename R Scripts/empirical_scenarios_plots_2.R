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
file_name1 = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), paste(1, ".pdf", sep=""), sep="")
file_name2 = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), paste(2, ".pdf", sep=""), sep="")
file_name3 = paste(paste(graph_name, gsub(" ", "_", scenario), sep="_"), paste(3, ".pdf", sep=""), sep="")


clean_ratios <- data$ratios[!is.na(data$ratios)]
clean_weights <- weights[!is.na(weights)]

pch_skip = as.numeric(args[9])

line_types <- c(1, 1, 2, 1, 2, 1, 2, 1)
pch_styles <- c(NA, 1, NA, NA, NA, NA, NA, NA)
line_width = 2

#for (name in names(weights)) {
#  if (length(grep("Random", name))>0) line_types <- c(line_types, 2)
#  if (length(grep("Informed", name))>0) line_types <- c(line_types, 1)
#}

print(" ++ Plotting activity: Step value 1")
if (format == "pdf") pdf(file_name1) else png(file_name1)
min_y = min(min(clean_weights), min(data$real_act_y))
max_y = max(max(clean_weights), max(data$real_act_y))
par(mar=c(5,5,1.5,1)+.1)
plot(weights$taus, weights$sim_act, type="l", pch=pch_styles[1], xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=line_types[1], lwd=line_width, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), xaxt="n")
axis(1, at=c(0, 2, 4, 6, 8, 10, 12), labels=c("Init", 2, 4, 6, 8, 10, 12), cex.axis=cex_paper)
#points(tail(weights$taus, n=1), tail(weights$sim_act, n=1), pch=1, col=colors[1], cex=cex_size)
lines(data$real_act_x, data$real_act_y, type="o", pch=pch_styles[2], lty=line_types[1], lwd=line_width, col=colors[2], cex=cex_size)
lines(weights$taus, weights[,3], type="l", pch=pch_styles[3], lty=line_types[3], lwd=line_width, col=colors[3], cex=cex_size)
lines(weights$taus, weights[,6], type="l", pch=pch_styles[4], lty=line_types[4], lwd=line_width, col=colors[4], cex=cex_size)
#points(tail(weights$taus, n=1), tail(weights[,i], n=1), pch=i, col=colors[i], cex=cex_size)

#legend("bottom", inset=c(0, -0.3), pch=seq(1, length(weights)+1, 1), col=colors, legend=legend_text, lty=1, cex=0.7, ncol=2)


#title(substitute(atop(scenario * " for " * bold(graph_name),
#                      "with " * rand_itas * " Random Iterations"),
#                 list(scenario = scenario, graph_name = graph_name, rand_itas = rand_itas)), cex.main=cex_paper)
grid(col="gray", lwd=line_width)
dev.off()

print(" ++ Plotting activity: Step value 2")
if (format == "pdf") pdf(file_name2) else png(file_name2)
min_y = min(min(clean_weights), min(data$real_act_y))
max_y = max(max(clean_weights), max(data$real_act_y))
par(mar=c(5,5,1.5,1)+.1)
plot(weights$taus, weights$sim_act, type="l", pch=pch_styles[1], xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=line_types[1], lwd=line_width, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), xaxt="n")
axis(1, at=c(0, 2, 4, 6, 8, 10, 12), labels=c("Init", 2, 4, 6, 8, 10, 12), cex.axis=cex_paper)
#points(tail(weights$taus, n=1), tail(weights$sim_act, n=1), pch=1, col=colors[1], cex=cex_size)
lines(data$real_act_x, data$real_act_y, type="o", pch=pch_styles[2], lty=line_types[1], lwd=line_width, col=colors[2], cex=cex_size)
lines(weights$taus, weights[,4], type="l", pch=pch_styles[5], lty=line_types[5], lwd=line_width, col=colors[5], cex=cex_size)
lines(weights$taus, weights[,7], type="l", pch=pch_styles[6], lty=line_types[6], lwd=line_width, col=colors[6], cex=cex_size)
#points(tail(weights$taus, n=1), tail(weights[,i], n=1), pch=i, col=colors[i], cex=cex_size)

#legend("bottom", inset=c(0, -0.3), pch=seq(1, length(weights)+1, 1), col=colors, legend=legend_text, lty=1, cex=0.7, ncol=2)


#title(substitute(atop(scenario * " for " * bold(graph_name),
#                      "with " * rand_itas * " Random Iterations"),
#                 list(scenario = scenario, graph_name = graph_name, rand_itas = rand_itas)), cex.main=cex_paper)
grid(col="gray", lwd=line_width)
dev.off()

print(" ++ Plotting activity: Step value 3")
if (format == "pdf") pdf(file_name3) else png(file_name3)
min_y = min(min(clean_weights), min(data$real_act_y))
max_y = max(max(clean_weights), max(data$real_act_y))
par(mar=c(5,5,1.5,1)+.1)
plot(weights$taus, weights$sim_act, type="l", pch=pch_styles[1], xlab=expression(tau ~ " (in months)"), ylab="Activity", lty=line_types[1], lwd=line_width, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), xaxt="n")
axis(1, at=c(0, 2, 4, 6, 8, 10, 12), labels=c("Init", 2, 4, 6, 8, 10, 12), cex.axis=cex_paper)
#points(tail(weights$taus, n=1), tail(weights$sim_act, n=1), pch=1, col=colors[1], cex=cex_size)
lines(data$real_act_x, data$real_act_y, type="o", pch=pch_styles[2], lty=line_types[1], lwd=line_width, col=colors[2], cex=cex_size)
lines(weights$taus, weights[,5], type="l", pch=pch_styles[7], lty=line_types[7], lwd=line_width, col=colors[7], cex=cex_size)
lines(weights$taus, weights[,8], type="l", pch=pch_styles[8], lty=line_types[8], lwd=line_width, col=colors[8], cex=cex_size)
#points(tail(weights$taus, n=1), tail(weights[,i], n=1), pch=i, col=colors[i], cex=cex_size)

#legend("bottom", inset=c(0, -0.3), pch=seq(1, length(weights)+1, 1), col=colors, legend=legend_text, lty=1, cex=0.7, ncol=2)


#title(substitute(atop(scenario * " for " * bold(graph_name),
#                      "with " * rand_itas * " Random Iterations"),
#                 list(scenario = scenario, graph_name = graph_name, rand_itas = rand_itas)), cex.main=cex_paper)
grid(col="gray", lwd=line_width)
dev.off()

pdf(paste(scenario, "_legend_2.pdf"), width=14, height=1)
par(mar=c(0,0,0,0))
plot.new()
legend("center", pch=pch_styles, col=colors, legend=legend_text, lty=line_types, lwd=1, cex=1, ncol=4)
dev.off