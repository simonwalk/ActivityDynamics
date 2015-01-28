setwd("/Users/simon/Desktop/Projects/DynamicNetworksResults/graph_binaries/ratios/")
setwd("/Volumes/DataStorage/Programming/DynamicNetworksResults/graph_binaries/ratios")


df = read.table("BEACHAPEDIA_ratios.txt", sep="\t", header=T)
df = read.table("APBR_ratios.txt", sep="\t", header=T)
df = read.table("CHARACTERDB_ratios.txt", sep="\t", header=T)
df = read.table("SMWORG_ratios.txt", sep="\t", header=T)
df = read.table("W15M_ratios.txt", sep="\t", header=T)
df = read.table("AARDNOOT_ratios.txt", sep="\t", header=T)
df = read.table("AUTOCOLLECTIVE_ratios.txt", sep="\t", header=T)
df = read.table("CWW_ratios.txt", sep="\t", header=T)
df = read.table("NOBBZ_ratios.txt", sep="\t", header=T)


names = c("BEACHAPEDIA", "APBR", "CHARACTERDB", "CWW", "NOBBZ", "StackOverflow", 
          "HistoryStackExchange", "EnglishStackExchange", "MathStackExchange")
for(name in names){
  print(name)
  df = read.table(paste(name, "_ratios.txt", sep=""), sep="\t", header=T)
  pdf(paste(name, ".pdf", sep=""))
  plot(df$agg_activity, type="b", pch=1, ylab="Activity", xlab="Month", col="blue", ylim=c(0, max(df$agg_activity)), 
       main=paste(name, sep=""))
  legend("topright", legend=c("Activity"), col=c("blue"), lty=c(1), pch=c(1))
  dev.off()
}
