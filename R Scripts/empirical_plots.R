args <- commandArgs(trailingOnly=T)
print(" ++ Setting work directory")
setwd(args[1])
print(" ++ Reading combined data")
data = read.table(args[2], sep="\t", header=T, colClasses = "numeric")
#print(" ++ Reading taus data")
#sim_act_x = read.table(args[4], header=F)
#print(" ++ Reading weights data")
#sim_act_y = read.table(args[3], header=F)
graph_name = paste(args[3], "static", sep="_")
mode = args[4]
format = args[5]
dtau = args[6]
mu = args[7]
ac = args[8]
k1 = args[9]
cex_paper = 1.5
cex_size = 1.5
colors = c("#000000", "#858585")
xlabel = substitute(tau ~ " (in " * mode * ")", list(mode=mode))
line_width = 2
print(paste(" ++ Plot Format: ", format, sep=""))

print(" ++ Plotting ratios and k1")
if (format == "pdf") pdf(paste(graph_name, "_ratios.pdf", sep="")) else png(paste(graph_name, "_ratios.png", sep=""))
clean_ratios <- data$ratios[!is.na(data$ratios)]
min_y = min(clean_ratios, as.numeric(k1))
max_y = max(clean_ratios, as.numeric(k1))
k1_x = seq(0,length(data$ratios),1)
k1_y = as.numeric(rep(k1, length(k1_x)))
par(mar=c(5,5,4,5)+.1)
plot(head(data$real_act_x, -1), head(data$ratios, -1), type="o", lwd=line_width, pch=4, col=colors[1], xlab=xlabel, ylab="Ratio", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(head(k1_x, -1), head(k1_y, -1), type="o", lwd=line_width, pch=1, lty=2, col=colors[2], cex=cex_size)
rho = sd(data$ratios, na.rm=TRUE)
rho_n = rho/as.numeric(k1)
rob = round(rho_n, digits=4)
title(substitute(atop("Ratio " ~ (frac(lambda,mu)) ~ " over " ~ tau ~ " (in " * mode * ")",
                      "for " ~ bold(network)),
                 list(network = args[3], mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
#legend("bottomright", pch=c(pchstyle,NA), col=colors, legend=c("Ratio", expression(kappa[1])), lty=c(1,2), cex=cex_paper)
dev.off()

print(" ++ Plotting ratio legend")
pdf("ratio_legend_static.pdf", width=14, height=1)
par(mar=c(0,0,0,0))
plot.new()
legend("center", pch=c(4,1), col=colors, legend=c("Ratio", expression(kappa[1])), lty=c(1,2), lwd=1, cex=1, ncol=2)
dev.off

print(" ++ Plotting activity")
if (format == "pdf") pdf(paste(graph_name, "_activity.pdf", sep="")) else png(paste(graph_name, "_activity.png", sep=""))
min_y = min(data$sim_act_y, data$real_act_y)
max_y = max(data$sim_act_y, data$real_act_y)
par(mar=c(5,5,4,5)+.1)
plot(data$real_act_x, data$sim_act_y, type="o", lwd=line_width, pch=4, xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type="o", lwd=line_width, cex=cex_size, lty=1, pch=1, col=colors[2])
title(substitute(atop("Activity over " ~ tau ~ " (in " * mode * ")",
                      "for " ~ bold(network)),
                 list(network=args[3], mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col="#858585", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)
#legend("topright", pch=c(NA,pchstyle), col=colors, legend=c("Simulated Activity", "Observed Activity"), lty=c(1,1), cex=cex_paper)
dev.off()

print(" ++ Plotting activity legend")
pdf("activity_legend_static.pdf", width=14, height=1)
par(mar=c(0,0,0,0))
plot.new()
legend("center", pch=c(4,1), col=colors, legend=c("Simulated Activity", "Empirical Activity"), lty=c(1,1), lwd=1, cex=1, ncol=2)
dev.off

print(" ++ Plotting Error of Simulation")
#sim_act_list <- list()
#i = 1
#while(i <= length(data$real_act_x)) {
#  #print(i)
#  closest_tau = which(abs(sim_act_x - data$real_act_x[i])==min(abs(sim_act_x - data$real_act_x[i])))
#  #print(closest_tau)
#  sim_act_list[length(sim_act_list) + 1] <- sim_act_y[closest_tau,]
#  i = i + 1
#}
#print(length(data$real_act_y))
#print(length(sim_act_list))
errors <- as.numeric(data$sim_act_y) - data$real_act_y
errors_x <- seq(0,length(errors) - 1,1)
#print(errors)
rmse <- sqrt(mean(errors^2))
#print(rmse)
if (format == "pdf") pdf(paste(graph_name, "_error.pdf", sep="")) else png(paste(graph_name, "_error.png", sep=""))
par(mar=c(5,5,4,5)+.1)
plot(errors_x, errors, type="o", lwd=line_width, xlab=xlabel, ylab="Activity", lty=1, pch=4, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
title(substitute(atop("Error of Simulation over " ~ tau ~ " (in " * mode * ")",
                      "for" ~ bold(network) ~ "(RMSE = " ~ rmse ~ ")"), list(network=args[3], rmse=round(rmse, 2), mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()