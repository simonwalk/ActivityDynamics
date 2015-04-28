args <- commandArgs(trailingOnly=T)
print(" ++ Setting work directory")
setwd(args[1])
print(" ++ Reading combined data")
data = read.table(args[2], sep="\t", header=T, colClasses = "numeric")
print(" ++ Reading taus data")
sim_act_x = read.table(args[4], header=F)
print(" ++ Reading weights data")
sim_act_y = read.table(args[3], header=F)
graph_name = paste(args[5], args[6], sep="_")
mode = args[6]
format = args[7]
dtau = args[8]
mu = args[9]
ac = args[10]
k1 = args[11]
cex_paper = 1.5
cex_size = 0.6
colors = c("#000000", "#858585")
xlabel = substitute(tau ~ " (in " * mode * ")", list(mode=mode))
if (args[6] == "days") {
  linetype = "l"
  pchstyle = NA
} else {
  linetype = "l"
  pchstyle = NA
}
print(paste(" ++ Plot Format: ", format, sep=""))

print(" ++ Plotting ratios and k1")
if (format == "pdf") pdf(paste(graph_name, "_ratios.pdf", sep="")) else png(paste(graph_name, "_ratios.png", sep=""))
clean_ratios <- data$ratios[!is.na(data$ratios)]
min_y = min(clean_ratios)
max_y = max(clean_ratios)
k1_x = seq(1,length(data$ratios),1)
k1_y = as.numeric(rep(k1, length(k1_x)))
plot(data$ratios, type=linetype, col=colors[1], xlab=xlabel, ylab="Ratio", cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y), pch=1)
lines(k1_x, k1_y, lty=2, col=colors[2])
rho = sd(data$ratios, na.rm=TRUE)
rho_n = rho/as.numeric(k1)
rob = round(rho_n, digits=4)
title(substitute(atop("Ratio " ~ (frac(lambda,mu)) ~ " over " ~ tau ~ " (in " * mode * ")",
                      Delta ~ tau == ~ dt ~ ", " ~ mu == ~ dp ~ ", " ~ kappa[1] == ~ k1 ~ ", " ~ rho == ~ rho_n),
                 list(dt = dtau, dp = mu, k1 = k1, rho_n=rob, mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
legend("bottomright", pch=c(pchstyle,NA), col=colors, legend=c("Ratio", expression(kappa[1])), lty=c(1,2), cex=cex_paper)
dev.off()

print(" ++ Plotting activity")
if (format == "pdf") pdf(paste(graph_name, "_activity.pdf", sep="")) else png(paste(graph_name, "_activity.png", sep=""))
min_y = min(min(sim_act_y), min(data$real_act_y))
max_y = max(max(sim_act_y), max(data$real_act_y))
par(mar=c(5,5,4,5)+.1)
plot(sim_act_x[,1], sim_act_y[,1], type="l", pch=4, xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(min_y, max_y))
lines(data$real_act_x, data$real_act_y, type=linetype, lty=1, pch=1, col=colors[2])
title(substitute(atop("Activity over " ~ tau ~ " (in " * mode * ")", 
                      Delta ~ tau == ~ dt ~ ", " ~ mu == ~ dp ~ ", " ~ a[c] == ~ ac),
                 list(dt = dtau, dp = mu, ac = ac, mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
#par(new=TRUE)
#plot(data$real_act_x, data$real_act_y, type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
#     col="#858585", cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
#axis(4, cex.axis=cex_paper)
#mtext("Real Activity",side=4,line=3, cex=cex_paper)
legend("topright", pch=c(NA,pchstyle), col=colors, legend=c("Simulated Activity", "Observed Activity"), lty=c(1,1), cex=cex_paper)
dev.off()

print(" ++ Plotting Error of Simulation")
sim_act_list <- list()
i = 1
while(i <= length(data$real_act_x)) {
  #print(i)
  closest_tau = which(abs(sim_act_x - data$real_act_x[i])==min(abs(sim_act_x - data$real_act_x[i])))
  #print(closest_tau)
  sim_act_list[length(sim_act_list) + 1] <- sim_act_y[closest_tau,]
  i = i + 1
}
#print(length(data$real_act_y))
#print(length(sim_act_list))
errors <- as.numeric(sim_act_list) - data$real_act_y
errors_x <- seq(0,length(errors) - 1,1)
#print(errors)
rmse <- sqrt(mean(errors^2))
#print(rmse)
if (format == "pdf") pdf(paste(graph_name, "_error.pdf", sep="")) else png(paste(graph_name, "_error.png", sep=""))
plot(errors_x, errors, type="l", xlab=xlabel, ylab="Activity", lty=1, col=colors[1], cex.axis=cex_paper, cex.lab=cex_paper)
title(substitute(atop("Error of Simulation over " ~ tau ~ " (in " * mode * ")",
                      "(RMSE = " ~ rmse ~ ")"), list(rmse=rmse, mode=mode)), cex.main=cex_paper)
grid(col="gray", lwd=1)
dev.off()