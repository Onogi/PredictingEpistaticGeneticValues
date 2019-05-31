library(ranger)
mse<-function(x,y){
  mean((x-y)^2)
}

Ntree<-10000
Mtry1<-c(4,14,20,40,60)
Mtry2<-c(2,4,8,16)
Nsim<-3

for(d in c(111, 112, 113)){
  Metrics<-matrix(0,Nsim,6)
  if(((d%/%10)%%10)==1) Mtry<-Mtry2 else Mtry<-Mtry1
  for(sim in 1:Nsim){
    cat(d,sim,"\n")
    Geno<-read.csv(paste("Data",d,".sim",sim,".Geno.train.csv",sep=""),row.names = 1)
    Pheno<-read.csv(paste("Data",d,".sim",sim,".Pheno.train.csv",sep=""),row.names = 1)
    Pheno.sd<-sd(Pheno$Pheno)
    Pheno.mean<-mean(Pheno$Pheno)
    Data<-data.frame(y=(Pheno$Pheno-Pheno.mean)/Pheno.sd,Geno)
    
    #cross validation in training data
    N<-nrow(Geno)
    Fold<-matrix(1:N,nc=5)
    Mse<-matrix(0,nr=length(Ntree),nc=length(Mtry))
    rownames(Mse)<-Ntree
    colnames(Mse)<-Mtry
    for(ntree in Ntree){
      for(mtry in Mtry){
        Prediction.cv<-NULL
        for(fold in 1:5){
          Result<-ranger(y~., data=Data[-Fold[,fold],], num.trees=ntree, mtry=mtry)
          Prediction.cv<-c(Prediction.cv, predict(Result, Data[Fold[,fold],])$predictions)
        }
        Mse[as.character(ntree),as.character(mtry)]<-mse(Prediction.cv,Data$y)
      }
    }
    pos<-which.min(Mse)
    if(pos%%length(Ntree)==0) {
      ntree<-Ntree[length(Ntree)]
      mtry<-Mtry[pos%/%length(Ntree)]
    } else {
      ntree<-Ntree[pos%%length(Ntree)]
      mtry<-Mtry[pos%/%length(Ntree)+1]
    }
    
    #fitted value
    Result<-ranger(y~., data=Data, num.trees=ntree, mtry=mtry)
    Fittedvalue<-predict(Result, Data)

    #prediction
    Geno.test<-read.csv(paste("Data",d,".sim",sim,".Geno.test.csv",sep=""),row.names = 1)
    Pheno.test<-read.csv(paste("Data",d,".sim",sim,".Pheno.test.csv",sep=""),row.names = 1)
    Data.test<-data.frame(y=(Pheno.test$Pheno-Pheno.mean)/Pheno.sd,Geno.test)
    Prediction<-predict(Result, Data.test)
    
    yhat.train<-Fittedvalue$predictions*Pheno.sd+Pheno.mean
    yhat.test<-Prediction$predictions*Pheno.sd+Pheno.mean
    Metrics[sim,]<-c(mse(Pheno$GV, yhat.train),
                     cor(Pheno$GV, yhat.train),
                     coef(lm(Pheno$GV~yhat.train))[2],
                     mse(Pheno.test$GV, yhat.test),
                     cor(Pheno.test$GV, yhat.test),
                     coef(lm(Pheno.test$GV~yhat.test))[2]
                     )
  }
  colnames(Metrics)<-c("MSE_train","Cor_train","Coef_train","MSE_test","Cor_test","Coef_test")
  write.csv(Metrics,paste("Metrics.Data",d,".sim1-",Nsim,".rf1.csv",sep=""))
}
