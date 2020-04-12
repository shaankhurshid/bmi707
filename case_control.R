# Dependencies
library(plyr)
library(data.table)

# Script to create cases/controls for accel project

# Load phenotype data
load(file='/Volumes/medpop_afib/skhurshid/accel/accel_pheno_041120.RData')

# Extra vars
accel_pheno[,':='(bmi_accel = (wt_accel/(ht_accel/100)**2))]

# Isolate to complete data
complete <- accel_pheno[complete_time==1]
  
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
match_list <- matcher(strata_cat=c("sex","race_binary","tobacco_accel",
                      "bpmed_accel","prev_mi_accel"),
             strata_numeric=c("age_accel","bmi_accel"),
             wiggle_numeric=c(1,5),
                      cases=complete[incd_af==1]$ID,
                      data=complete)
cases <- match_list[[1]]
controls <- match_list[[2]]

# Match the unmatched (n=8)
matched <- cases[!is.na(controls$ID)]
unmatched <- cases[is.na(controls$ID)]
unused <- complete[!(ID %in% matched$ID) & !(ID %in% controls$ID)]

match_list2 <- matcher(strata_cat=c("sex","race_binary","tobacco_accel",
                                   "bpmed_accel","prev_mi_accel"),
                      strata_numeric=c("age_accel","bmi_accel"),
                      wiggle_numeric=c(5,10),
                      cases=unmatched$ID,
                      data=unused)

controls2 <- match_list2[[2]]

# Replace the unmatched in the original tables
controls[is.na(controls$ID)] <- controls2

# QC on matched variables
mean(cases$age_accel); sd(cases$age_accel); mean(controls$age_accel); sd(controls$age_accel)
mean(cases$bmi_accel); sd(cases$bmi_accel); mean(controls$bmi_accel); sd(controls$bmi_accel)
count(cases$sex); count(controls$sex)
count(cases$race_binary); count(controls$race_binary)
count(cases$tobacco_accel); count(controls$tobacco_accel)
count(cases$bpmed_accel); count(controls$bpmed_accel)
count(cases$prev_mi_accel); count(controls$prev_mi_accel)

## linker file for IDs
linker <- fread('/Volumes/medpop_afib/skhurshid/accel/7089_17488_linker.csv')
# Join to linker
setkey(cases,ID); setkey(controls,ID); setkey(linker,app7089)
cases[linker,":="(lubitz_id = i.app17488)]
controls[linker,":="(lubitz_id = i.app17488)]

# Save outputs
all <- rbind(cases,controls)
write.csv(cases,file='/Volumes/medpop_afib/skhurshid/bmi707/cases.csv')
write.csv(controls,file='/Volumes/medpop_afib/skhurshid/bmi707/controls.csv')
write.csv(all,file='/Volumes/medpop_afib/skhurshid/bmi707/cases_controls.csv')


