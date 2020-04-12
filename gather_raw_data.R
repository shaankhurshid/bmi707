# Script to format raw acceleration files for DL
library(stringr)
library(data.table)

# Threads
setDTthreads(12)

# Load IDs of interest
list <- fread(file='/mnt/ml4cvd/projects/skhurshid/bmi707/cases_controls.csv')

# Function
flatten <- function(key){
out <- list()
index <- as.numeric(substr(list.files('/mnt/ml4cvd/projects/skhurshid/accel_raw/'),0,7))
  for (i in 1:length(key)){
  if(!(key[i] %in% index)){print(paste0("I could not find ID # ",key[i],'. Skipping')); next}
  data <- fread(paste0('/mnt/ml4cvd/projects/skhurshid/accel_raw/',key[i],'_90004_0_0.csv'))
  out[[i]] <- as.numeric(unlist((data[,1])))[1:112320]
  if(i %% 50 == 0){print(paste0('Just finished flattening ',i,' out of ',length(key),' files!'))}
  }
return(do.call(rbind,out))
}

# save acceleration output
output <- flatten(key=list$lubitz_id)
write.table(output,'/mnt/ml4cvd/projects/skhurshid/bmi707/accel_flat.tsv',row.names = F,
            col.names = F,sep='\t')

# save key for ordering
write.table(list$ID,'/mnt/ml4cvd/projects/skhurshid/bmi707/key.tsv',row.names = F,
col.names = F,sep='\t')

# save outcome for analysis
write.table(list$incd_af,'/mnt/ml4cvd/projects/skhurshid/bmi707/outcome.tsv',row.names = F,
            col.names = F,sep='\t')

