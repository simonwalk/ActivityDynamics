args <- commandArgs(trailingOnly=T)
print(" ++ Setting new wd")
setwd(args[1])
print(" ++ Reading combined data")
data = read.table(args[2], header=T, sep="\t", colClasses = "numeric")
print(" ++ Reading taus data")
sim_act_x = read.table(args[4], header=F)
print(" ++ Reading weights data")
sim_act_y = read.table(args[3], header=F)
graph_name = paste(args[5], paste(args[6], "epochs", sep="_"), sep="_")
mode = args[6]
format = args[7]
cex_paper = 1.5
cex_size = 0.6
colors = c("#000000", "#858585", "#FF0000")
xlabel = substitute(tau ~ " (in " * mode * ")", list(mode=mode))
xvalues <- seq(0,length(data$ratios) - 1,1)
if (args[6] == "days") {
  linetype = "l"
  pchstyle = NA
} else {
  linetype = "l"
  pchstyle = NA
}

print(" ++ Plotting acs")
if (format == "pdf") pdf(paste(graph_name, "_acs.pdf", sep="")) else png(paste(graph_name, "_acs.png", sep=""))
min_y = min(data$a_cs)
max_y = max(data$a_cs)
plot(data$a_cs, type=linetype, col=colors[1], xlab=xlabel, ylab=expression("a"[c]), cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
title("Critical Activity Evolution", cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()

print(" ++ Plotting ratios and k1s")
if (format == "pdf") pdf(paste(graph_name, "_ratios.pdf", sep="")) else png(paste(graph_name, "_ratios.png", sep=""))
clean_ratios <- data$ratios[!is.na(data$ratios)]
min_y = min(min(clean_ratios), min(data$k1s))
max_y = max(max(clean_ratios), max(data$k1s))
par(mar=c(5,5,4,5)+.1)
plot(xvalues, data$ratios, type=linetype, col=colors[1], xlab=xlabel, ylab="Ratio", cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
lines(xvalues, data$k1s, col=colors[2], lty=2)
title(substitute("Ratio " ~ (frac(lambda,mu)) ~ " over " ~ tau ~ " (in " * mode * ")", list(mode=mode)))
legend("bottomright", pch=c(pchstyle,NA), col=colors, legend=c("Ratio", expression(kappa[1])), lty=c(1,2), cex=cex_paper)
grid(col="gray", lwd=1)
dev.off()

print(" ++ Plotting gs")
if (format == "pdf") pdf(paste(graph_name, "_gs.pdf", sep="")) else png(paste(graph_name, "_gs.png", sep=""))
min_y = min(data$gs)
max_y = max(data$gs)
plot(data$gs, type=linetype, col=colors[1], xlab=xlabel, ylab=expression("g(a"[j] ~ ")"), cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
title(expression("g(a"[j] ~ ")"), cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()

print(" ++ Plotting max_qs")
if (format == "pdf") pdf(paste(graph_name, "_max_qs.pdf", sep="")) else png(paste(graph_name, "_max_qs.png", sep=""))
min_y = min(data$max_qs)
max_y = max(data$max_qs)
plot(data$max_qs, type=linetype, col=colors[1], xlab=xlabel, ylab="q", cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
title("Maximal Peer Activity Flow Evolution", cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()

print(" ++ Plotting mus")
if (format == "pdf") pdf(paste(graph_name, "_mus.pdf", sep="")) else png(paste(graph_name, "_mus.png", sep=""))
min_y = min(data$mus)
max_y = max(data$mus)
plot(data$mus, type=linetype, col=colors[1], xlab=xlabel, ylab=expression(mu), cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
title(expression("Evolution of " ~ mu), cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()

print(" ++ Plotting activity (debug)")
if (format == "pdf") pdf(paste(graph_name, "_activity_debug.pdf", sep="")) else png(paste(graph_name, "_activity_debug.png", sep=""))
min_y = min(min(sim_act_y), min(data$real_act_y))
max_y = max(max(sim_act_y), max(data$real_act_y))
par(mar=c(5,5,4,5)+.1)
plot(sim_act_x[,1], sim_act_y[,1], type="l", pch=4, xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type=linetype, lty=1, pch=1, col=colors[2])
title(substitute("Activity over " ~ tau ~ " (in " * mode * ")", list(mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col=colors[2], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)
legend("topright", pch=c(NA,pchstyle), col=colors, legend=c("Simulated Activity", "Observed Activity"), lty=c(1,1), cex=cex_paper)
dev.off()

print (" ++ Plotting activity")
if (format == "pdf") pdf(paste(graph_name, "_activity.pdf", sep="")) else png(paste(graph_name, "_activity.png", sep=""))
min_y = min(min(data$sim_act_y), min(data$real_act_y))
max_y = max(max(data$sim_act_y), max(data$real_act_y))
par(mar=c(5,5,4,5)+.1)
plot(data$sim_act_x, data$sim_act_y, type="l", pch=4, xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type=linetype, lty=1, pch=1, col=colors[2])
lines(data$fil_act_x, data$fil_act_y, type=linetype, lty=1, pch=1, col=colors[3])
title(substitute("Activity over " ~ tau ~ " (in " * mode * ")", list(mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
legend("top", pch=c(NA,pchstyle), col=colors, legend=c("Simulated Activity", "Observed Activity", "Filtered Activity"), lty=c(1,1,1), cex=1)
dev.off()

print(" ++ Plotting Error of Simulation")
#sim_act_list <- list()
#i = 1
#while(i <= length(data$real_act_x)) {
  #print(i)
  #closest_tau = which(abs(sim_act_x - data$real_act_x[i])==min(abs(sim_act_x - data$real_act_x[i])))
  #print(closest_tau)
  #sim_act_list[length(sim_act_list) + 1] <- sim_act_y[closest_tau,]
  #i = i + 1
#}
#print(length(data$real_act_y))
#print(length(sim_act_list))
#errors <- as.numeric(sim_act_list) - data$real_act_y
errors <- data$sim_act_y - data$real_act_y
errors_x <- seq(0,length(errors) - 1,1)
#print(errors)
rmse <- sqrt(mean(errors^2))
#print(rmse)
if (format == "pdf") pdf(paste(graph_name, "_error.pdf", sep="")) else png(paste(graph_name, "_error.png", sep=""))
plot(errors_x, errors, type="l", xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex.axis=cex_paper, cex.lab=cex_paper)
title(substitute(atop("Error of Simulation over " ~ tau ~ " (in " * mode * ")",
                      "(NRMSE = " ~ rmse ~ ")"), list(rmse=rmse, mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()