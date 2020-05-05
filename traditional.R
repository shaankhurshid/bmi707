# Script to perform traditional stats on accel data

# Dependencies
library(pROC)

#################################### Step 1: BMI analyses
############ Part A: CNN
# Load outputs from BMI model
bmi_actual <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/val_actual_bmi.csv',header=TRUE)
bmi_predicted <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/val_preds_bmi.csv',header=TRUE)

# Check correlation
cor.test(bmi_predicted$`0`, bmi_actual$`0`, method = "pearson") #r=0.116, p=0.03

# Scatterplot
pdf(file='/Volumes/medpop_afib/skhurshid/bmi707/bmi_scatter.pdf',height=3,width=3,pointsize=4)
plot(bmi_predicted$`0`,bmi_actual$`0`,pch=19,col='#3182bd',xlab='',ylab='',bty='n',
     xlim=c(min(bmi_predicted$`0`),28),ylim=c(min(bmi_actual$`0`),60),cex=0.6)
axis(2,at=seq(0,60,10))
axis(1,at=seq(18,28,2))
mtext(expression(paste("Predicted BMI (kg/m"^"2",")")),1,line=2.8)
mtext(expression(paste("Actual BMI (kg/m"^"2",")")),2,line=2.3)
segments(0,0,60,60,lty=5)

dev.off()

############ Part B: linear model using average
# Gather values for these individuals for non-DL analysis
## Holdout
holdout_controls <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_controls.csv')
holdout_cases <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/holdout_cases.csv')

holdout_value <- c(holdout_controls$value,holdout_cases$value)
holdout_value <- data.frame(value=holdout_value)

## Train
train_controls <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls.csv')
train_cases <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_cases.csv')

train_controls_value <- train_controls$value
train_cases_value <- train_cases$value
train_value <- c(train_controls_value,train_cases_value)
train_value <- data.frame(value=train_value)

train_bmi <- c(train_controls$bmi_accel,train_cases$bmi_accel)

## Fit linear model
linear_model <- lm(train_bmi ~ value,data=train_value)
predicted_bmi_linear <- predict(linear_model,newdata=holdout_value)

cor.test(predicted_bmi_linear, bmi_actual$`0`, method = "pearson") #r=0.329, p<0.01

# Scatterplot
pdf(file='/Volumes/medpop_afib/skhurshid/bmi707/bmi_scatter_linear.pdf',height=3,width=3,pointsize=4)
plot(predicted_bmi_linear,bmi_actual$`0`,pch=19,col='#3182bd',xlab='',ylab='',bty='n',
     xlim=c(min(predicted_bmi_linear),32),ylim=c(min(bmi_actual$`0`),60),cex=0.6)
axis(2,at=seq(0,60,10))
axis(1,at=seq(20,32,2))
mtext(expression(paste("Predicted BMI (kg/m"^"2",")")),1,line=2.8)
mtext(expression(paste("Actual BMI (kg/m"^"2",")")),2,line=2.3)
segments(0,0,60,60,lty=5)

dev.off()

#################################### Step 2: AF analyses
############ Part A: CNN
# Load outputs from AF model
af_predicted <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/af_cnn_preds.csv')[2:347,2]
af_predicted_binary <- ifelse(af_predicted >= 0.50,1,0)
af_actual <- c(rep(0,173),rep(1,173))
  
# Check overall accuracy
acc <- length(af_predicted_binary[af_predicted_binary[,1]==af_actual])/347

############ Part B: logistic model using average
# Gather values for these individuals for non-DL analysis
## Truth
y_test <- c(rep(0,173),rep(1,173))
y_train <- c(rep(0,694),rep(1,694))

## Values
train_controls <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_controls.csv')
train_cases <- fread(file='/Volumes/medpop_afib/skhurshid/bmi707/train_cases.csv')

x_train <- data.frame(value=c(train_controls$value,train_cases$value))
x_test <- data.frame(value=c(holdout_controls$value,holdout_cases$value))

## Fit logistic model
logistic_model <- glm(y_train ~ value,data=x_train,family='binomial') # p=0.035
predicted_af_logistic <- predict(logistic_model,newdata=x_test,type='response')
predicted_af_logistic_binary <- ifelse(predicted_af_logistic<0.50,0,1)

logistic_roc <- roc(response=y_test,predictor=predicted_af_logistic_binary)
cnn_roc <- roc(response=af_actual,predictor=af_predicted_binary)

# ROC
pdf('/Volumes/medpop_afib/skhurshid/bmi707/af_roc.pdf',pointsize=6,
    height=3,width=3)
par(oma=c(2,2,1,1))
par(mar=c(2,2,1,1))

# ROC lines
plot(logistic_roc,xlim=c(0.8,0.2),col='#3182bd',axes=FALSE,xlab='',ylab='',lwd=2,identity=FALSE)
plot(cnn_roc,xlim=c(0.8,0.2),col='#f03b20',axes=FALSE,xlab='',ylab='',lwd=2,identity=FALSE,add=TRUE)

# Identity line
segments(1,0,0,1,col='black',lwd=2,lty=4)

# Axes
axis(1,at=seq(0,1,0.1),cex.axis=1.8,pos=0)
axis(2,at=seq(0,1,0.1),cex.axis=1.8,pos=1,las=1)
mtext(side=1,'1 - Specificity',cex=2.2,line=1)
mtext(side=2,'Sensitivity',cex=2.2,line=1)

legend(0.58,0.275,lty=1,lwd=3,c('Logistic (AUC 0.523)','CNN (AUC 0.549)'),
       col=c('#3182bd','#f03b20'),bty='n',cex=1)

dev.off()