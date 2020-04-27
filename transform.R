# Script to create a holdout set for BMI707 model

# Load dataset
## Key for cases/controls
all <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/cases_controls.csv')
setnames(all,"V1","subject_id")

## Load raw accel data
raw <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/accel_flat.tsv')
raw[,':='(subject_id = 1:nrow(raw))]
setcolorder(raw,'subject_id')

## Transform accel data
set.seed(1)
factor <- runif(112120,min=0.8,max=1.2)

transformed <- list()
for (i in 1:nrow(raw)){
  transformed[[i]] <- raw[i,2:length(raw)]*factor
  if(i %% 50 == 0){print(paste0('Just finished transforming ',i,' out of ',nrow(raw),' files!'))}
}
transformed <- do.call(rbind,transformed)
transformed$subject_id <- 1:nrow(transformed)
setcolorder(transformed,'subject_id')

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

# Join on transformed data
setkey(transformed,subject_id); setkey(holdout_cases,subject_id); setkey(holdout_controls,subject_id); 
setkey(train_cases,subject_id); setkey(train_controls, subject_id)

holdout_controls_transformed <- transformed[subject_id %in% holdout_controls$subject_id]
holdout_cases_transformed <- transformed[subject_id %in% holdout_cases$subject_id]
train_cases_transformed <- transformed[subject_id %in% train_cases$subject_id]
train_controls_transformed <- transformed[subject_id %in% train_controls$subject_id]

# Save out
write.table(factor,file='/Volumes/medpop_afib/skhurshid/bmi707/transformation_factor.csv',sep=',', col.names = F, row.names = F)
write.table(holdout_controls_transformed,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls_transformed.csv',sep=',', col.names = F, row.names = F)
write.table(holdout_cases_transformed,file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_cases_transformed.csv',sep=',', col.names = F, row.names = F)
write.table(train_cases_transformed,file='/Volumes/medpop_afib/skhurshid/bmi707/train_cases_transformed.csv',sep=',', col.names = F, row.names = F)
write.table(train_controls_transformed,file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls_transformed.csv',sep=',', col.names = F, row.names = F)