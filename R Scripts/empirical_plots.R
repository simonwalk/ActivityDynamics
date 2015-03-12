#__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
#__license__ = "GPL"
#__version__ = "0.0.1"
#__email__ = "simon.walk@tugraz.at"
#__status__ = "Development"

args <- commandArgs(trailingOnly=T)
print(" ++ Setting work directory")
setwd(args[1])
print(" ++ Reading weights table")
t = read.table(args[2], sep="\t", header=F, fill=T)
t <- t(t)
dtau = args[3]
dpsi = args[4]
ac = args[5]
k1 = args[6]
#intrinsic = read.table(args[8], sep="\t", header=F, fill=T)
#extrinsic = read.table(args[9], sep="\t", header=F, fill=T)
#intrinsic <- abs(intrinsic)
print(" ++ Plotting data (ratios)")
pdf(paste(args[7], "_ratio.pdf", sep=""))
max_y_l = max(t[,2], na.rm=T)
max_x_b = max(t[,1], na.rm=T)
max_y_r = max(t[,4], na.rm=T)
max_x_t = max(t[,3], na.rm=T)
cex_size = 0.6

par(mar=c(5,4,4,1)+.1)
c1 <- rgb(0.5, 0.5, 0.5, alpha=1)
c2 <- rgb(0,0,0,alpha=1)
c3 <- rgb(0.8, 0.4, 0.3, alpha=0.9)
cex_paper = 1.5

plot(t[,1], t[,2], type="l", pch=1, xlab=expression(tau ~ " (in months)"), ylab="Ratio", 
     col=c1, cex=cex_size, xlim=c(0, max_x_t), cex.axis=cex_paper, cex.lab=cex_paper)

print(" ++ Plotting data (activity)")

x = seq(0,max_x_t,1)
y = as.numeric(rep(k1, length(x)))
rho = sd(t[,2], na.rm=TRUE)
rho_n = rho/as.numeric(k1)
#rob = var(t[,2], na.rm=TRUE)
#rob = rob/as.numeric(k1)
#act = sum(t[,6], na.rm=TRUE)
#months = max(t[,5], na.rm=T)
#lmact  = t[,6][months+1]
#mean_act = act/months
#act_inertia = 1/rho
#act_mass = 1/rho * act
#mean_momentum = act_mass * mean_act
#curr_momentum = act_mass * lmact

rob = round(rho_n, digits=4)
title(substitute(atop("Ratio " ~ (frac(lambda,mu)) ~ " over " ~ tau ~ " (in months)",
                      Delta ~ tau == ~ dt ~ ", " ~ mu == ~ dp ~ ", " ~ kappa[1] == ~ k1 ~ ", " ~ rho == ~ rho_n),
                 list(dt = dtau, dp = dpsi, k1 = k1, rho_n=rob)), cex.main=cex_paper)
lines(x, y, lty=2, col=c2)
grid(col="gray", lwd=1)
legend("bottomright", pch=c(1,NA), col=c(c1, c2), legend=c("Ratio", expression(kappa[1])), lty=c(1,2), cex=cex_paper)
dev.off()
#y_min = min(intrinsic[,1], min(extrinsic[,1]))
#y_max = max(intrinsic[,1], extrinsic[,1]+t[,4], t[,4])
pdf(paste(args[7], "_activity.pdf", sep=""))
par(mar=c(5,5,4,5)+.1)
plot(t[,3], t[,4], type="l", pch=4,xlab=expression(tau ~ " (in months)"), 
     ylab=expression("Simulated Activity over " ~ a[c]), lty=2,
     col=c2, cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)#, ylim=c(y_min, y_max))
title(substitute(atop("Activity over " ~ tau ~ " (in months)", 
                      Delta ~ tau == ~ dt ~ ", " ~ mu == ~ dp ~ ", " ~ a[c] == ~ ac),
                 list(dt = dtau, dp = dpsi, ac = ac)), cex.main=cex_paper)
grid(col="gray", lwd=1)
par(new=TRUE)
plot(t[,5], t[,6], type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
     col=c1, cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
axis(4, cex.axis=cex_paper)
mtext("Real Activity",side=4,line=3, cex=cex_paper)
legend("topleft", pch=c(NA,2), col=c(c2, c1), legend=c("Simulated Activity", "Observed Activity"), lty=c(2,1), cex=cex_paper)
dev.off()
