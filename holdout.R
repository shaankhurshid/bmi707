# Script to create a holdout set for BMI707 model

# Dependencies
library(data.table)

# Load dataset
## Key for cases/controls
all <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/cases_controls.csv')
setnames(all,"V1","subject_id")

## Load raw accel data
raw <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/accel_flat.tsv')
raw[,':='(subject_id = 1:nrow(raw))]
setcolorder(raw,'subject_id')

## Generate holdout set
set.seed(99)
cases <- all[incd_af==1]
controls <- all[incd_af==0]
holdout_cases <- cases[sample(nrow(cases),size=173)]

## Find their controls
# Function to match cases/controls
matcher <- function(strata_cat,strata_numeric,cases,data,
                    key='ID',wiggle_numeric){
  out_case <- list(); out_ctrl <- list()
  controls <- data[!(get(key) %in% cases)]
  for (i in 1:length(cases)){
    out_case[[i]] <- data[get(key)==cases[i]]
    ctrl <- controls
    for (j in 1:length(strata_numeric)){
      ctrl <- ctrl[abs(get(strata_numeric[j]) - out_case[[i]][[strata_numeric[j]]]) <= wiggle_numeric[j]]
    }
    for (k in 1:length(strata_cat)){
      ctrl <- ctrl[get(strata_cat[k])==out_case[[i]][[strata_cat[k]]]]
      if (k==length(strata_cat)){
        out_ctrl[[i]] <- ctrl[1]
        controls <- controls[!get(key) %in% out_ctrl[[i]]]
        if (i %% 50 == 0){print(paste0("Matched ",i,' cases!'))}}
    }
  }
  case <- do.call(rbind,out_case)
  ctrl <- do.call(rbind,out_ctrl)
  return(list(case,ctrl))
}

# Initial run
other_cases <- cases[!(ID %in% holdout_cases$ID)]
control_pool <- all[!(ID %in% other_cases$ID)]

match_list <- matcher(strata_cat=c("sex","race_binary","tobacco_accel",
                                   "bpmed_accel","prev_mi_accel"),
                      strata_numeric=c("age_accel","bmi_accel"),
                      wiggle_numeric=c(1,5),
                      cases=holdout_cases$ID,
                      data=control_pool)

holdout_cases <- match_list[[1]]
holdout_controls <- match_list[[2]]

# Match the unmatched (n=1)
matched <- holdout_cases[!is.na(holdout_controls$ID)]
unmatched <- holdout_cases[is.na(holdout_controls$ID)]
unused <- control_pool[!(ID %in% matched$ID) & !(ID %in% holdout_controls$ID)]

match_list2 <- matcher(strata_cat=c("sex","race_binary","tobacco_accel",
                                    "bpmed_accel","prev_mi_accel"),
                       strata_numeric=c("age_accel","bmi_accel"),
                       wiggle_numeric=c(5,10),
                       cases=unmatched$ID,
                       data=unused)

controls2 <- match_list2[[2]]

# Replace the unmatched in the original tables
holdout_controls[is.na(holdout_controls$ID)] <- controls2

# Designate training data
train_cases <- cases[!(ID %in% holdout_cases$ID)]
train_controls <- controls[!(ID %in% holdout_controls$ID)]

# Join on raw data
setkey(raw,subject_id); setkey(holdout_cases,subject_id); setkey(holdout_controls,subject_id); 
setkey(train_cases,subject_id); setkey(train_controls, subject_id)

holdout_controls_data <- raw[subject_id %in% holdout_controls$subject_id]
holdout_cases_data <- raw[subject_id %in% holdout_cases$subject_id]
train_cases_data <- raw[subject_id %in% train_cases$subject_id]
train_controls_data <- raw[subject_id %in% train_controls$subject_id]

# BMI data
holdout_controls_bmi <- all[subject_id %in% holdout_controls_data$subject_id]$bmi_accel
holdout_cases_bmi <- all[subject_id %in% holdout_cases_data$subject_id]$bmi_accel
train_controls_bmi <- all[subject_id %in% train_cases_data$subject_id]$bmi_accel
train_cases_bmi <- all[subject_id %in% train_controls_data$subject_id]$bmi_accel

# Save out
write.table(holdout_controls_data,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls_data.csv',sep=',', col.names = F, row.names = F)
write.table(holdout_cases_data,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_cases_data.csv',sep=',', col.names = F, row.names = F)
write.table(train_cases_data,file='/Volumes/medpop_afib/skhurshid/bmi707/train_cases_data.csv',sep=',', col.names = F, row.names = F)
write.table(train_controls_data,file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls_data.csv',sep=',', col.names = F, row.names = F)
write.csv(holdout_controls,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls.csv')
write.csv(holdout_cases,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_cases.csv')

# Save out
write.table(holdout_controls_bmi,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls_bmi.csv',sep=',', col.names = F, row.names = F)
write.table(holdout_cases_bmi,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_cases_bmi.csv',sep=',', col.names = F, row.names = F)
write.table(train_controls_bmi,file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls_bmi.csv',sep=',', col.names = F, row.names = F)
write.table(train_cases_bmi,file='/Volumes/medpop_afib/skhurshid/bmi707/train_cases_bmi.csv',sep=',', col.names = F, row.names = F)
