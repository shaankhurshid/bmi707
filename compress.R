# Script to downsample raw accel data

# Load data
holdout_controls <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls_data.csv',header = F)
holdout_cases <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls_data.csv',header = F)
train_cases <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls_data.csv',header = F)
train_controls <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls_data.csv',header = F)

# Function to downsample
downsample <- function(data,step){
out <- list()
set <- list()
n <- 1
  for (i in 1:nrow(data)){
    j <- 1
    while (j < length(data)){
      full <- data[i]
      set[[n]] <- mean(full[j:j+step])
      n <- n+1
      j <- j+step
      if(j==length(data)){out[[i]] <- unlist(set)}
    }}}

a <- downsample(data=train_cases[1,2:length(train_cases)],step=10)

