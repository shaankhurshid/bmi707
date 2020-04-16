# Script for processing accelerometer phenotypes for possible BMI 707 project

# Dependencies
library(data.table)
library(plyr)
library(survival)
library(prodlim)

# Load data
# Pre-processed phenotype data
load(file='/Volumes/medpop_afib/skhurshid/accel/phenos_031320.RData')
withdrawals <- fread('/Volumes/medpop_afib/skhurshid/PRS/withdrawals_020420.csv') # UKBB withdrawals
acceleration <- fread('/Volumes/medpop_afib/skhurshid/accel/ukbb_accel2.csv')
activity <- fread('/Volumes/medpop_afib/skhurshid/accel/ukbb_activity_all.csv')
enroll <- fread('/Volumes/medpop_afib/skhurshid/accel/censor_dates.csv')

# Instance 1 data
sbp1 <- fread('/Volumes/medpop_afib/skhurshid/accel/sbp_instance1.csv')
dbp1 <- fread('/Volumes/medpop_afib/skhurshid/accel/dbp_instance1.csv')
ht1 <- fread('/Volumes/medpop_afib/skhurshid/accel/height_instance1.csv')
wt1 <- fread('/Volumes/medpop_afib/skhurshid/accel/weight_instance1.csv')
tob1 <- fread('/Volumes/medpop_afib/skhurshid/accel/tobacco_instance1.csv')
med1 <- fread('/Volumes/medpop_afib/skhurshid/accel/med_instance1.csv')

################################################################ Step 0: Cleanup
# Update phenotype data (instance 1 is prior to all accel data, so we can use it for accel analyses)
# Decode med/tob phenotypes prior to merge
tob1[,':='(tob = ifelse(value==0,"Never",
                        ifelse(value==1,"Previous",
                               ifelse(value==2,"Current",NA))))]
med1[,':='(bp_med = ifelse(value==2,1,0))]

# Average the BPs prior to merge
sbp_avg <- sbp1[,sbp_avg := mean(value),by=sample_id]
dbp_avg <- dbp1[,dbp_avg := mean(value),by=sample_id]

# Add to phenotype table
setkey(sbp1,sample_id); setkey(dbp1,sample_id); setkey(ht1,sample_id); setkey(wt1,sample_id); 
setkey(tob1,sample_id); setkey(med1,sample_id)

phenos[sbp1,':='(sbp_instance1 = i.sbp_avg)]
phenos[dbp1,':='(dbp_instance1 = i.dbp_avg)]
phenos[ht1,':='(ht_instance1 = i.value)]
phenos[wt1,':='(wt_instance1 = i.value)]
phenos[med1,':='(bpmed_instance1 = i.bp_med)]
phenos[tob1,':='(tob_instance1 = i.tob)]

## Set non-answers to NA (-1 = Don't know, and -3 = Prefer not to answer)
names <- c('duration_walks','duration_moderate','duration_vigorous','freq_walk','freq_mod','freq_vigorous')
for (j in (names)){set(activity,i=which(activity[[j]] %in% c(-1,-3)),j=j,value=NA)}

## Generate MET summary columns
activity[,':='(met_walk = duration_walks*freq_walk*3.3,
               met_mod = duration_moderate*freq_mod*4,
               met_vigorous = duration_vigorous*freq_vigorous*8)]

activity[!c(is.na(met_walk) & is.na(met_mod) & is.na(met_vigorous)),
         total_met := apply(.SD,1,sum,na.rm=T),.SDcols=c('met_walk','met_mod','met_vigorous')]

# Activity data
## Set non-answers to NA (-1 = Don't know, and -3 = Prefer not to answer)
names <- c('duration_walks','duration_moderate','duration_vigorous','freq_walk','freq_mod','freq_vigorous')
for (j in (names)){set(activity,i=which(activity[[j]] %in% c(-1,-3)),j=j,value=NA)}

## Generate MET summary columns
activity[,':='(met_walk = duration_walks*freq_walk*3.3,
               met_mod = duration_moderate*freq_mod*4,
               met_vigorous = duration_vigorous*freq_vigorous*8)]

activity[!c(is.na(met_walk) & is.na(met_mod) & is.na(met_vigorous)),
         total_met := apply(.SD,1,sum,na.rm=T),.SDcols=c('met_walk','met_mod','met_vigorous')]

# Acceleration data
## Date cleanup
for (j in (c('start_date','end_date'))){set(acceleration,j=j,value=strtrim(acceleration[[j]],width=10))}
for (j in (c('start_date','end_date'))){set(acceleration,j=j,value=as.Date(acceleration[[j]],format='%Y-%m-%d'))}

# Enroll data
## Date cleanup
for (j in ('enroll_date')){set(enroll,j=j,value=as.Date(enroll[[j]],format='%Y-%m-%d'))}

## Time to accel variable
setkey(enroll,'sample_id'); setkey(acceleration,'sample_id')
acceleration[enroll,enroll_date := i.enroll_date]
acceleration[,survyears_accel := as.numeric(end_date - enroll_date)/365.25]


################################################################ Step 1: MI?
# Exclusions
## Remove revoked consent
no_revoked <- phenos[!(ID %in% withdrawals$V1)] # N=123
## Remove individuals with prevalent MI
no_prevmi <- no_revoked[prevalent_Myocardial_Infarction!=1] # N=12074
## Remove individuals with < 14 days follow-up or no FU
accel <- no_prevmi[c(!is.na(survyears_Myocardial_Infarction) & (survyears_Myocardial_Infarction*365.25 >= 14))] # N=37

################################################################ Step 2: Join with raw accel data
# Recommended QC for accelerometer data (N=103691)
## Could not calibrate
no_calibrate <- acceleration[calibration==1] #N=6996
## Inadequate wear time
acceleration <- no_calibrate[wear_time==1] #N=4

# Perform join
setkey(acceleration,sample_id); setkey(accel,ID)
accel_pheno <- accel[acceleration,nomatch=0] # Lose 2016 for no clinical data

# Friendly names
setnames(accel_pheno,'survyears_Myocardial_Infarction','mi.t')
setnames(accel_pheno,'incident_Myocardial_Infarction','incd_mi')

# Remove people who developed MI prior to accelerometer (N=641)
accel_pheno <- accel_pheno[c(incd_mi==0 | (incd_mi== 1 & mi.t >= survyears_accel))]
