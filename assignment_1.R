# Dependencies
library(data.table)
library(plyr)

# Script to double check answers to assignment 1 for BMI 707
train_data <- fread('https://www.dropbox.com/s/n0jjos6faoyqncp/diabetesTrain.csv?dl=1')
test_data <- fread('https://www.dropbox.com/s/6uu2754iqv4cxei/diabetesTest.csv?dl=1')

# Linear model associating BMI and DM outcome
mod <- glm(Outcome ~ BMI,family='binomial',data=train_data)

# Linear model associating all X and BMI outcome
mod <- glm(Outcome ~ Pregnancies + Glucose + BloodPressure +
             SkinThickness + Insulin + BMI + DiabetesPedigreeFunction +
             Age,family='binomial',data=train_data)

summary(mod)
