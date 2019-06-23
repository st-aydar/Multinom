rm(list = ls())

library(readxl)
library(dplyr)
library(nnet)
library(treemap)
library(gridExtra)
library(grid)
library(data.table)
library(cluster)
library(factoextra)
library(ggplot2)
library(rpart)
library(partykit)
library(caret)
library(forecast)
library(randomForest)

setwd("E:/Cloud/TU Cloud/Cloud/FU Cloud/Multivariate Methods/Multinom")
data <- read_xlsx("data.xlsx")
var <- read_xlsx("data_char.xlsx")

colnames(data)[c(2, 6, 11, 18, 20)] <- c("Bez", "BezNr", "Waehler", "Linke", "Gruene")
colnames(var)[c(2, 5, 9, 10, 18, 21, 23, 25, 27, 29, 31, 33, 35, 37, 47, 49, 51, 53, 57)] <-
  c("Bez", "BezNr", "einwhn", "kind6", "auslder", "alter1825", "alter2535", "alter3545", "alter4560", "alter6070",
    "alter70", "weib", "migrHint", "ledig", "evan", "kath", "wlg0", "wlg1", "entwk")


data <- group_by(data, BezNr)
var <- group_by(var, BezNr)

data <- summarise(data, Bez = mean(as.numeric(Bez)), CDU = sum(CDU), SPD = sum(SPD), LINKE = sum(Linke), GRUENE = sum(Gruene), 
                    AfD = sum(AfD), FDP = sum(FDP), Sonstige = sum(Sonstige), 
                    n = CDU + SPD + LINKE + GRUENE + AfD + FDP + Sonstige,
                    CDUp = CDU / n, SPDp = SPD / n, LINKEp = LINKE / n, GRUENEp = GRUENE / n,
                    AfDp = AfD / n, FDPp = FDP / n, Sonstigep = Sonstige / n)

ergb <- c()
for (i in 1:nrow(data)) {
  ergb[i] <- colnames(data)[which.max(data[i, 11:17]) + 10]
}
data$ergb <- ergb
head(data)

var <- summarise(var, Bez = mean(as.numeric(Bez)), n = sum(einwhn), kind = sum(kind6) / n, 
                  auslder = sum(auslder) / n, alt1 = sum(alter1825) / n,
                  alt2 = sum(alter2535) / n, alt3 = sum(alter3545) / n, 
                  alt4 = sum(alter4560) / n, alt5 = sum(alter6070) / n, 
                  alt6 = sum(alter70) / n, gndr = sum(weib) / n, 
                  mig = sum(migrHint) / n, ledig = sum(ledig) / n, 
                  christ = (sum(evan) + sum(kath)) / n, wlg0 = sum(wlg0) / n, 
                  wlg1 = sum(wlg1) / n, entwk = sum(entwk) / n)
head(var)


# spliting the data set
set.seed(1)
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)

train <- cbind(data[sample, ], var[sample, -c(1, 2, 3)])
test  <- cbind(data[-sample, ], var[-sample, -c(1, 2, 3)])

# model selection / estimation
data$ergb <- relevel(as.factor(data$ergb), ref = "CDUp")
varName <- colnames(var)[-c(1, 2)]
model <- multinom(paste0("ergb ~ ", paste(varName, collapse = " + ")), data = train,
                     maxit = 1000)
  # model <- step(model)
modelSum <- summary(model)

z <- modelSum$coefficients / modelSum$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2

# prediction
test$predict <- predict(model, type = "class", newdata = test)
sum(test$ergb == test$predict) / nrow(test)

test$ergb <- as.factor(test$ergb)

vplayout <- function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2, 1)))

tmPlot(test, index = c("Bez", "BezNr"), vSize = "n", vColor = "ergb", type = "categorical",
       vp = vplayout(1, 1))
tmPlot(test, index = c("Bez", "BezNr"), vSize = "n", vColor = "predict", type = "categorical",
       vp = vplayout(2, 1))
# Making one big table out of two
row.names(data) <- c(data$BezNr)
row.names(var) <- c(var$BezNr)
oneBigTable <- cbind(data, var)
colnames(oneBigTable)[24:29] <- c("alter1825", "alter2535", "alter3545", "alter4560", "alter6070", "alter70")
str(oneBigTable)

# Random Forest

# Split data in two partitions (70% Training, 30% Validation)
set.seed(10)
Daten <- oneBigTable[,22:36]
Daten <- cbind(Daten, oneBigTable$ergb)
str(Daten)
inTrain <- createDataPartition(Daten$`oneBigTable$ergb`, p = 0.7, list = FALSE)
train <- Daten[inTrain, ]
vali <- Daten[-inTrain,]


# Train a Random Forest with the default parameters using pclass & title
rf.train.7 <- train[, 1:15]
rf.label <- as.factor(train$`oneBigTable$ergb`)

# Train a Random Forest using pclass, title, parch, & family.size

set.seed(1234)
rf.7 <- randomForest(x = rf.train.7, y = rf.label, importance = TRUE, ntree = 1000)
rf.7
varImpPlot(rf.7, main="Importance of the Factors")
#-------------------------------------
# Random forest with classification with all variables
set.seed(1)
train <- na.omit(train)
fit<-randomForest(Daten$`oneBigTable$ergb`~.,data=Daten,mtry=sqrt(12), importance =TRUE,subset=inTrain)
fit

# Importance of each variable
importance((fit))

# Plot the importance of each variable
varImpPlot(fit,main="Importance of the Factors")

# Nicer plot
test<-sort(importance(fit)[,1])/max(importance(fit)[,1])
test<-data.frame(x1=labels(test),y1=test)
test<-transform(test, x1 = reorder(x1, y1))

ggplot(data=test, aes(x=x1, y=y1)) + 
  ylab("Mean Decrease Gini") + xlab("") +
  geom_bar(stat="identity",fill="skyblue",alpha=.8,width=.75) + 
  coord_flip()

# Compare the performance of the bagged decision tree on the training and validation data
pred_t_ranFor<-predict(fit,newdata=train)
pred_v_ranFor<-predict(fit,newdata=vali)

confusionMatrix(pred_t_ranFor,train$`oneBigTable$ergb`)
confusionMatrix(pred_v_ranFor,vali$`oneBigTable$ergb`)

# Distance by voters

df1 <- transpose(data[,3:7]) # Use df as shorter name
rownames(df1) <- colnames(data)[c(3:7)]
View(df1)

df2 <- df1
df2 <- na.omit(df2)
df2 <- scale(df2)

set.seed(123)
ss <- sample(1:50,15) # Take 15 random columns
df2 <- df1[ss,] # Subset the 15 rows
df2.scaled <- scale(df1)

dist.eucl <- dist(df2.scaled, method = "euclidean")
round(as.matrix(dist.eucl)[1:3, 1:3], 1)


########################################################

#A data frame with 18 observations on 8 variables:

data(flower)
head(flower, 3)
str(flower)
dd <- daisy(flower) #gower
round(as.matrix(dd)[1:3, 1:3], 2)
fviz_dist(dist.eucl) #The color level is proportional to the value of the dissimilarity between observations
#Red: high similarity, Blue: low similarity