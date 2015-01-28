#setwd("/Volumes/StorageDisk/DynamicNetworks/graph_sources/weights/")
#setwd("/Volumes/MyBook4TB/DynamicNetworks/graph_sources/weights/")
setwd("/Users/simon/Desktop/Projects/DynamicNetworksResults/graph_sources/weights/")

require(graphics)

graph_names <- c("Karate_0PERC_RAND", "Random_MEDIUM_0PERC_RAND", "PrefAttach_MEDIUM_0PERC_RAND", "SBM_ASC_SAME_CONN_0PERC_RAND", "SBM_DEG_CORR_0PERC_RAND")
legend_labels <- c("Karate", "Random", "PrefAttach", "SBM ASC", "SBM DC")

ratios = c(1)
rseq = seq(10, 120, 10)
ratios = append(ratios, rseq)
print(ratios)

ratio_strings = c("10", "100", "200", "300", "400", "500", "600", "700", "800", "900", "1000", "1100", "1200")

deltatau = "0001"
iterations = "10000"

ratio_counter = 1
min_x = 0
max_x = 0
max_y = 0

# get min_x and max_x
for(ratio in ratio_strings){
  filepaths = c()
  for(gn in graph_names){
    filepaths = append(filepaths, paste(gn, "/", gn, "_", iterations, "_iterations_", ratio, "_", deltatau, "_mean_per_ita.txt", sep=""))
  }  
  # get max_x and max_y
  for(fp in filepaths) {
    t <- read.table(fp, sep="\t")
    mx = length(t$V1) * 0.001
    my = max(t$V1)    
    x <- seq(length(t$V1)-1)
    if(max_x < mx){
      max_x = mx
    }
    if(max_y < my){
      max_y = my
    }
  }
}


pdf("../Average_Activity_per_NW_at_Convergence.pdf")
x_val = ratios
ita = 1
for(gn in graph_names){
  line <- c()
  filepaths = c()
  for(rat in ratio_strings){
    filepaths = append(filepaths, paste(gn, "/", gn, "_", iterations, "_iterations_", rat, "_", deltatau, "_mean_per_ita.txt", sep=""))
  }
  for(fp in filepaths) {
    t <- read.table(fp, sep="\t")
    #print(fp)
    t <- t[nrow(t),]
    #print(t$V1)
    line <- append(line, t$V1)
  }
  colors=rainbow(5)
  
  # plot values

  if(ita==1){
    t <- bquote("Average Activity per Network at Convergence for all ratios")
    par(mar=c(5.1, 4.1, 4.1, 8.1))
    plot(rev(x_val), rev(line), xlim=c(max(ratios), 1), ylim=c(0, max_y), type="b", pch=1, col=colors[ita],
         xlab=bquote("Ratio of " ~ frac(lambda,mu)), ylab="Average Activity per Network at Convergence", 
         main=t)
  } else {
    lines(rev(x_val), rev(line), type="b", pch=ita, col=colors[ita])
  }
  ita = ita+1
  
  legend("topright", inset=c(-0.32, 0), legend=c(legend_labels), pch=c(1,2,3,4,5), col=colors, lty=c(1,2,3,4,5), xpd=T)
  #par(op)
  grid(lwd = 2)
}
dev.off()
