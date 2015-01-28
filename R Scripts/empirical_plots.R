args <- commandArgs(trailingOnly=T)
print(" ++ Setting new wd")
setwd(args[1])
print(" ++ Reading table")
t = read.table(args[2], sep="\t", header=F, fill=T)
t <- t(t)
dtau = args[3]
dpsi = args[4]
ac = args[5]
k1 = args[6]

print(args[8])
print(args[9])

intrinsic = read.table(args[8], sep="\t", header=F, fill=T)
extrinsic = read.table(args[9], sep="\t", header=F, fill=T)
intrinsic <- abs(intrinsic)
#colnames(t) <- t[0,]
print(" ++ plotting data")
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

y = as.numeric(rep(k1, max_x_b+1))
x = seq(0,max_x_t+1,1)
rob = var(t[,2], na.rm=TRUE)
print(rob)
rob = rob/as.numeric(k1)
print(rob)
title(substitute(atop("Ratio " ~ (frac(lambda,mu)) ~ " over " ~ tau ~ " (in months)", 
                      Delta ~ tau == ~ dt ~ ", " ~ Delta ~ psi == ~ dp ~ ", " ~ kappa[1] == ~ k1 ~ ", " ~ rho == ~ rob),  
                 list(dt = dtau, dp = dpsi, k1 = k1, rob=round(rob, digits=4))), cex.main=cex_paper)

lines(x, y, lty=2, col=c2)
grid(col="gray", lwd=1)
#rect(0, k1, max_x_b, as.numeric(k1)*0.98, col=rgb(0.8, 0.8, 0.8, alpha=0.2), border="transparent") 
legend("bottomright", pch=c(1,NA), col=c(c1, c2), legend=c("Ratio", expression(kappa[1])), lty=c(1,2), cex=cex_paper)
dev.off()

y_min = min(intrinsic[,1], min(extrinsic[,1]))
y_max = max(intrinsic[,1], extrinsic[,1]+t[,4], t[,4])
#print(t[,4])
#print(t[,6])

#print(length(intrinsic[,1]))
#print(length(extrinsic[,1]))

pdf(paste(args[7], "_activity.pdf", sep=""))
par(mar=c(5,5,4,5)+.1)
plot(t[,3], t[,4], type="l", pch=4,xlab=expression(tau ~ " (in months)"), 
     ylab=expression("Simulated Activity over " ~ a[c]), 
     col=c2, cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper, ylim=c(y_min, y_max))
lines(t[,3], intrinsic[,1], col="red", type="l", lty=3)
lines(t[,3], extrinsic[,1] + t[,4], col="blue", type="l", lty=4)
#lines(t[,3], t[,4] - extrinsic[,1] + intrinsic[,1], col="purple")
#lines(t[,3], extrinsic[,1]+intrinsic[,1], col="green")

title(substitute(atop("Activity over " ~ tau ~ " (in months)", 
                      Delta ~ tau == ~ dt ~ ", " ~ Delta ~ psi == ~ dp ~ ", " ~ a[c] == ~ ac),  
                 list(dt = dtau, dp = dpsi, ac = ac)), cex.main=cex_paper)
grid(col="gray", lwd=1)
par(new=TRUE)
plot(t[,5], t[,6], type="o", lty=1, pch=2,xaxt="n",yaxt="n",xlab="",ylab="", 
     col=c1, cex=cex_size, cex.axis=cex_paper, cex.lab=cex_paper)
axis(4, cex.axis=cex_paper)
mtext("Real Activity",side=4,line=3, cex=cex_paper)
legend("topleft", pch=c(NA,2, NA,NA), col=c(c2, c1,"red","blue"), legend=c("Simulated Activity", "Real Activity", "Intrinsic Activity Decay", "Extrinsic Peer Influence"), lty=c(2,1,3,4), cex=cex_paper)
dev.off()

print(" ++ closing files")
