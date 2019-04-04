#!/usr/local/bin/Rscript

# Script to build model and make predictions for Driven Data Rinse over Run competition

# By: mlearn
# Date: March 2019

# Version: 0acd12b211fa503483d3567761b136cfc95764fe31fc9cfd3844795f260d5aed

# data.table for data loading and feature computation
library(data.table)
# vtreat for coding categorical variables
library(vtreat)
# lightgbm for modelling
library(lightgbm)


# read data
labels<-fread("../data/train_labels.csv")
f<-fread("../data/train_values.csv") #,nrows=100000) # abbreviate files for testing
testf<-fread("../data/test_values.csv") #,nrows=1000)
recipe<-fread("../data/recipe_metadata.csv")

# convert to phase string ordered factors and remove final_rinse from training data
f[,phase_factor:=factor(phase,levels = c("pre_rinse","caustic","intermediate_rinse","acid","final_rinse"))]
f[,phase_num:=as.numeric(phase_factor)]
f<-f[phase_num<5]
testf[,phase_factor:=factor(phase,levels = c("pre_rinse","caustic","intermediate_rinse","acid","final_rinse"))]
testf[,phase_num:=as.numeric(phase_factor)]
gc()

# Augment the training data with weights to match the generation process of the competition.
# For each process we create four processes with weights (phase_weight) to correspond to the 
# probability of stopping observation after each phase.
# We create "new_process_id" to index these augmented processes.  
f1<-f[phase_num<=1][,new_process_id:=paste("p1",process_id,sep=".")][,phase_weight:=0.1]
f2<-f[phase_num<=2][,new_process_id:=paste("p2",process_id,sep=".")][,phase_weight:=0.3]
f3<-f[phase_num<=3][,new_process_id:=paste("p3",process_id,sep=".")][,phase_weight:=0.3]
f4<-f[phase_num<=4][,new_process_id:=paste("p4",process_id,sep=".")][,phase_weight:=0.3]
fmatch<-rbind(f1,f2,f3,f4)
rm(f1,f2,f3,f4)
rm(f)
gc()
fmatch[,target_time_period:=NULL]

# produce summaries of numeric features including
# .1, .3, .5 (median), .7, .9 quantiles (odd deciles)
# mean, min, max, standard deviation (sd)
# mean of last {5, 25, 125, 625} observations
dosummarynum<-function(x) {
  return(list(median=median(x),
              q1=quantile(x,0.1),
              q9=quantile(x,0.9),
              mean=mean(x),
              min=min(x),
              max=max(x),
              meanlast5=mean(tail(x,5)),
              meanlast25=mean(tail(x,25)),
              meanlast125=mean(tail(x,125)),
              meanlast625=mean(tail(x,625)),
              q3=quantile(x,0.3),
              q7=quantile(x,0.7),
              sd=sd(x)))
}
ftrainnum<-fmatch[,c(list(process_id=process_id[1],phase_weight=phase_weight[1]),
                     unlist(lapply(.SD,dosummarynum),recursive = F)),
                  new_process_id,
                  .SDcols=c('supply_flow', 'supply_pressure', 'return_temperature',
                            'return_conductivity', 'return_turbidity','return_flow',
                            'tank_level_pre_rinse',  'tank_level_caustic', 'tank_level_acid',
                            'tank_level_clean_water', 'tank_temperature_pre_rinse', 'tank_temperature_caustic',
                            'tank_temperature_acid', 'tank_concentration_caustic',  'tank_concentration_acid')]
ftestnum<-testf[,unlist(lapply(.SD,dosummarynum),recursive = F),
                process_id,
                .SDcols=c('supply_flow', 'supply_pressure', 'return_temperature',
                          'return_conductivity', 'return_turbidity','return_flow',
                          'tank_level_pre_rinse',  'tank_level_caustic', 'tank_level_acid',
                          'tank_level_clean_water', 'tank_temperature_pre_rinse', 'tank_temperature_caustic',
                          'tank_temperature_acid', 'tank_concentration_caustic',  'tank_concentration_acid')]

# produce summary of Boolean features as a simple mean
dosummarybool<-function(x) {
  return(list(mean=mean(x)))
}
ftrainbool<-fmatch[,c(list(process_id=process_id[1]),
                      unlist(lapply(.SD,dosummarybool),recursive = F)),
                   new_process_id,
                   .SDcols=names(which(sapply(fmatch,is.logical)))]
ftestbool<-testf[,unlist(lapply(.SD,dosummarybool),recursive = F),
                 process_id,
                 .SDcols=names(which(sapply(fmatch,is.logical)))]

# add extra features for supply_flow as that feature seems to be important
# We add more deciles and the median of the supply flow in each phase
ftrainsupply<-fmatch[,list(process_id=process_id[1],
                           supply_flow_q6=quantile(supply_flow,0.6),
                           supply_flow_q7=quantile(supply_flow,0.7),
                           supply_flow_q8=quantile(supply_flow,0.8),
                           supply_flow_median_p1=median(supply_flow[phase_num==1]),
                           supply_flow_median_p2=median(supply_flow[phase_num==2]),
                           supply_flow_median_p3=median(supply_flow[phase_num==3]),
                           supply_flow_median_p4=median(supply_flow[phase_num==4])),new_process_id]
ftestsupply<-testf[,list(supply_flow_q6=quantile(supply_flow,0.6),
                         supply_flow_q7=quantile(supply_flow,0.7),
                         supply_flow_q8=quantile(supply_flow,0.8),
                         supply_flow_median_p1=median(supply_flow[phase_num==1]),
                         supply_flow_median_p2=median(supply_flow[phase_num==2]),
                         supply_flow_median_p3=median(supply_flow[phase_num==3]),
                         supply_flow_median_p4=median(supply_flow[phase_num==4])),process_id]

# summarise how long each phase we observe is and extract basic metadata (pipeline and object)
dosummaryphase<-function(x) {
  myt<-tabulate(x,4)
  names(myt)<-paste("phase",1:4,sep=".")
  return(as.list(myt))
}
ftrainmeta<-fmatch[,c(list(object_id=as.character(object_id[1]),pipeline=pipeline[1],
                           process_id=process_id[1]),
                      dosummaryphase(phase_num)),new_process_id]
ftestmeta<-testf[,c(list(object_id=as.character(object_id[1]),pipeline=pipeline[1]),
                    dosummaryphase(phase_num)),process_id]

# use vtreat to code pipeline and object (and incidentally phase lengths).  Objects are high-cardinality.
metaandtargets<-labels[ftrainmeta,,on="process_id"][,process_id:=NULL][,logtarget:=log(final_rinse_total_turbidity_liter)]
treatmentsN = designTreatmentsN(metaandtargets,setdiff(colnames(metaandtargets),
                                                       c("new_process_id","final_rinse_total_turbidity_liter")),
                                'logtarget')
ftrainmetaprep <- prepare(treatmentsN,ftrainmeta,pruneSig=c(),scale=TRUE)
ftestmetaprep <- prepare(treatmentsN,ftestmeta,pruneSig=c(),scale=TRUE)

# merge all features together
ftrainmetanum<-cbind(ftrainnum,ftrainmetaprep)
ftestmetanum<-cbind(ftestnum,ftestmetaprep)
ftrain<-recipe[ftrainmetanum,,on="process_id"][ftrainbool,,on=c("process_id","new_process_id")][ftrainsupply,,on=c("process_id","new_process_id")]
ftest<-recipe[ftestmetanum,on="process_id"][ftestbool,,on="process_id"][ftestsupply,,on="process_id"]
ftrainlabelled<-labels[ftrain,,on="process_id"][order(process_id)]

rm(fmatch)
gc()

# create feature matrix for lightgbm
xcols<-setdiff(colnames(ftrainlabelled),c("final_rinse_total_turbidity_liter","process_id",
                                    "new_process_id","phase_weight"))
trainx<-ftrainlabelled[,xcols,with=F]
testx<-ftest[,xcols,with=F]

# training target
trainy<-ftrainlabelled$final_rinse_total_turbidity_liter

# create a weight to cover three things:
# 1. the weights of the augmentation process
# 2. weights to covert a custom MAPE problem into an MAE problem that lightgbm can fit
# 3. downscale weights so that leaf weights in lightgbm works
trainweight<-ftrainlabelled$phase_weight/pmax(trainy,290000)*1e6

# form lightgbm dataset.  We downscale the target to make lightgbm work.
train<-lgb.Dataset(data=as.matrix(trainx),label=trainy/1e6,weight=trainweight)

# params for lightgbm
params <- list(objective = "regression_l1",
               metric = "l1",
               num_leaves = 800,
               nthread = 2
)

# fit the model
model <- lgb.train(params,
                   train,
                   21000,
                   list(train=train),
                   min_data = 80,
                   learning_rate = 0.001,
                   early_stopping_rounds = 300,
                   eval_freq = 50)

# create predictions (with upscaling of the target to match how we fed the data to lightgbm)
tosub<-data.frame(process_id=ftest$process_id,
                  final_rinse_total_turbidity_liter=predict(model,as.matrix(testx))*1e6)

# write output
write.csv(tosub,"tosubmit.csv",quote=F,row.names=F)

