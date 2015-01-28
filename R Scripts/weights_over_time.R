require(graphics)

args <- commandArgs(trailingOnly=T)
print(" ++ Setting new wd")
setwd(args[1])
print(" ++ Reading table")
t = read.table(args[2], sep="\t", header=F, fill=T)
#print(t)
xlimit = dim(t)[1]
limit = dim(t)[2]
x_vals <- as.numeric(t[1,])[1:xlimit-1]
t <- subset(t[2:limit,])
ratio = as.numeric(args[3])
dtau = as.numeric(args[4])
k1 = as.numeric(args[6])
max_y = max(t, na.rm=T)
y1 = as.numeric(t[,1])[!is.na(t[,1])]
num_nodes = length(as.numeric(t[1,])[!is.na(t[1,])])
cols = rainbow(num_nodes)

print(" ++ Plotting")

cex_paper=1.5
pdf(paste(args[5], "/", args[5], "_ratio_", ratio, ".pdf", sep=""))
plot(x_vals[1:length(y1)+1], y1, xlim=c(0, max(x_vals)), type="l", ylim=c(0,max_y), xlab=expression(tau), 
     ylab="Activity", col=cols[1], cex.axis=cex_paper, cex.lab=cex_paper)
title(substitute(atop("Activity per Node over " ~ tau, (frac(lambda, mu)==ratio~", "~Delta ~ tau == dtau ~ ", " ~ kappa[1] == k1)), 
                 list(ratio=ratio, dtau=dtau, k1=k1)), cex.main=cex_paper)

for(i in seq(2:num_nodes)){
  y = as.numeric(t[,i])[!is.na(t[,i])]
  x = x_vals[1:length(y)+1]
  lines(x,y, col=cols[i])
}
grid(col="gray", lwd=1)
dev.off()
print(" ++ Done")