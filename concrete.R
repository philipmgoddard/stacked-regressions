# stacked regression experiment with nnet models
# use concrete data from AppliedPredictiveModeling package
#################################################

# load packages
library(AppliedPredictiveModeling)
library(caret)
library(dplyr)
library(reshape2)
library(corrplot)
library(FUNctions)
library(nnet)
library(doMC)

# register parallel backend
registerDoMC(4)

# load data
data("concrete")
summary(concrete)

#################################################
# data exploration
#################################################

# check for missing data
vapply(concrete[, 1:9], function(x) sum(is.na(x)), numeric(1))

# what about duplicated predictive inputs?
# remove full duplicates (i.e. where outcome duplicated as well)
concrete <- concrete[!duplicated(concrete[, 1:9]), ]

# which just have duplicated mixes (input not outcome?)
concrete[duplicated(concrete[, 1:8]), ]

# here is an experiment for what is irreducible error:
# remove those that are full duplicates (e.g outcome as well)
# and assess standard error in compressive strength
concreteIrreducible <- concrete %>%
  group_by(Cement,
           BlastFurnaceSlag,
           FlyAsh,
           Water,
           Superplasticizer,
           CoarseAggregate,
           FineAggregate,
           Age) %>% 
  summarise(stderr = sd(CompressiveStrength) / sqrt(n()))

# we see that those which have same mixtures have on average
# a standard error of 3.89...
# gives some intuition of what the uncertainty in compressive
# strength is (and therefore irreducible model error)
mean(concreteIrreducible$stderr, na.rm = TRUE)

# average the outcome for duplicate mixtures 
concrete <- concrete %>% 
  group_by(Cement,
           BlastFurnaceSlag,
           FlyAsh,
           Water,
           Superplasticizer,
           CoarseAggregate,
           FineAggregate,
           Age) %>%
  summarise(CompressiveStrength = mean(CompressiveStrength))

# cast back to data.frame as tbl_df no good for caret
concrete <- data.frame(concrete)

# seperate input and outcome
input <- concrete[, 1:8]
outcome <- concrete[, 9]

# train-test split
set.seed(1234)
inTrain <- createDataPartition(outcome, p = 0.7, list = FALSE)
trainInput <- input[inTrain, ]
testInput <- input[-inTrain,]
trainOutcome <- outcome[inTrain]
testOutcome <- outcome[-inTrain]

# set plotting theme for lattice
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

# visualise relationshop between predictive inputs and outcome
featurePlot(x = trainInput,
            y = trainOutcome,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            layout = c(4, 2))

# visualise distributions of predictive inputs
inputPlots <- ggplotListDensR(trainInput)
ggMultiplot(inputPlots, 3)

# investigate between-predictor correlations
splom(trainInput)

# not too worried about correlations
corr <- cor(trainInput)
corrplot(corr, order = 'hclust')

#################################################
# cool, quick and easy setup as data quite nice.
# next, lets generate our 'level one' data

# APM book tells us for single layer nnet, 27 units are best
# train 20 nnets - use caret

# set training options
ctrl <-trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5)

# define training parameter 'grid' (constant tuning parameters here)
nnetGrid <- expand.grid(decay = 0.001, 
                        size = 27)

# fit models, store object in list for easy retrieval
models <- lapply(seq(1, 20), function(x) {
  set.seed(x)
  train(x = trainInput,
        y = trainOutcome,
        method = "nnet",
        tuneGrid = nnetGrid,
        preProc = c("center", "scale"),
        linout = TRUE,
        trace = FALSE,
        maxit = 1000,
        trControl = ctrl)
})

# can save models with this syntax to save re-runnning!
#save(models, file = "~/Desktop/nnetModels.RData")

# investigate resamples during training
resamp <- resamples(models)

# could do bwplot(resamp), but only want to show rmse...
resampRMSE <- as.data.frame(resamp, metric = resamp$metric[1])
meltResampRMSE <- melt(resampRMSE)

# visualise resamples
ggplot(meltResampRMSE, aes(x = variable, y = value), groups = variable) +
  geom_boxplot(fill = philTheme()[1],
    alpha = 0.5,
    lwd = 0.5) +
  scale_color_manual(values = philTheme()) +
  scale_fill_manual(values = philTheme()) +
  coord_flip() +
  theme_bw() +
  xlab("") + 
  ylab("RMSE") 

# now want to split the test set into a validation and test set
set.seed(1234)
inVal <- createDataPartition(testOutcome, p = 0.5, list= FALSE)
valInput <- testInput[inVal, ]
testInput <- testInput[-inVal, ]
valOutcome <- testOutcome[inVal]
testOutcome <- testOutcome[-inVal]

#################################################
# combining models
#################################################

# lets start with simple averaging approach

# simple averaging - validation set
valPred <- vapply(models, function(x) predict(x, newdata = valInput), numeric(148))

avgVal <- data.frame(predict = rowMeans(valPred),
                     actual = valOutcome)

caret::RMSE(avgVal$predict, avgVal$actual)
caret::R2(avgVal$predict, avgVal$actual)

# simple averaging - test set
testPred <- vapply(models, function(x) predict(x, newdata = testInput), numeric(148))

avgTest <- data.frame(predict = rowMeans(testPred),
                      actual = testOutcome)

caret::RMSE(avgTest$predict, avgTest$actual)
caret::R2(avgTest$predict, avgTest$actual)

# residuals
residData <- testInput
residData$predict <- avgTest$predict
residData$actual <- testOutcome
residData$resid <- residData$predict - residData$actual

# unfortunately not many samples in test set... but no obvious
# heterosketadicity
ggplot(residData, aes(predict, resid)) +
  geom_point(alpha = 0.5, cex = 2) + 
  theme_bw() + 
  geom_abline(intercept = 0, slope = 0, alpha = 0.7, linetype = "dashed") +
#  scale_color_manual(values = philTheme()) +
  xlab("Predicted Compressive Strength") + 
  ylab("Residual") + 
  theme(text = element_text(size = 18))

ggplot(residData, aes(predict, actual)) +
  geom_point(alpha = 0.5, cex = 2) + 
  theme_bw() + 
  geom_abline(intercept = 0, slope = 1, alpha = 0.7, linetype = "dashed") +
  #  scale_color_manual(values = philTheme()) +
  xlab("Predicted Compressive Strength") + 
  ylab("Actual Compressive Strength") + 
  theme(text = element_text(size = 18))

#################################################
# stacking
# breiman suggests: lm can perform sub optimal as correlated
# penalised (e.g. ridge better)
# also mentions that no-negativiy constraint works well
featurePlot(x = valPred,
            y = valOutcome,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"))

#################################################
# start by combining with a linear model

set.seed(1234)
linearStack <- train(x = valPred,
                     y = valOutcome,
                     method = "lm",
                     trControl = ctrl)

lmStacked <- data.frame(predict = predict(linearStack, testPred),
                        actual = testOutcome)

# not as good as simple averaging
caret::RMSE(lmStacked$predict, lmStacked$actual)
caret::R2(lmStacked$predict, lmStacked$actual)

#################################################
# penalised linear model with elastic net

enetGrid <- expand.grid(lambda = seq(0, 0.1, 0.01),
                        fraction = seq(0.05, 1, length = 20))

set.seed(1234)
enetStack <- train(x = valPred,
                   y = valOutcome,
                   method = "enet",
                   tuneGrid = enetGrid,
                   trControl = ctrl)

enetStackVal <- data.frame(predict = predict(enetStack, testPred),
                           actual = testOutcome)

# better than linear stack, competitive with averaging
# larger test set would be needed to more accurately assess
# which technique generates the best predictions
caret::RMSE(enetStackVal$predict, enetStackVal$actual)
caret::R2(enetStackVal$predict, enetStackVal$actual)

#################################################
# breiman advocates minimising the difference between
# the outcome and weighted sum of the 'level 1' data
# with no negativity. the weights dont neccessarily have to 
# add to one, but he mentions that little difference seen 
# if choose to enforce or not


# This function returns the RMSE of the ensemble based upon
# the weights for the predictions of each model.
# At the moment enforce sum to 1
# Note that we take as input weights for number of models -1
# as we can determine the last weight from all the others 
weightedStack <- function(modelWeights, trainPred, trainOutcome) {

  nPred <- length(modelWeights) + 1
  
  # check proportions in correct range
  for(i in 1:(nPred - 1)) {
    if(modelWeights[i] < 0 | modelWeights[i] > 1) return(10^38)
  }
  
  # deduce the final weight
  modelWeights <- c(modelWeights, 1 - sum(modelWeights))
  
  # this will be violated if the sum of the weights
  # doesnt add to one
  if(modelWeights[nPred] < 0 ) return(10^38)
  
  # we want this 'composite' SSE 
  return(sum(trainOutcome - sum(modelWeights * trainPred))^2)
}


# wrapper function around optim, using the cost function
# defined above
optimWeights <- function(trainPred, trainOutcome) {
  
  # how many models did we use to make predictions?
  nPred <- ncol(trainPred)
  
  # randomly build up vector of n weights
  # which add to 1. Build up nStart of these
  nStart <- 50
  weights <- vector('list', length = nStart)
  
  for(i in 1:nStart) {
    draws <- as.list(-log(runif(nPred))) 
    denom <- Reduce(sum, draws) 
    weights[[i]] <- simplify2array(Map(function(x, y) x / y,
                                       draws,
                                       denom)
    )
  }
  
  weights <- do.call(rbind.data.frame, weights)
  for (i in 1:ncol(weights)) names(weights)[i] <- i
  weights$SSE <- NA
  
  # loop over starting values and optimise for
  # minimum SSE for each of these
  # Use SANN or Nelder-Mead as discontinuous function
  for( i in 1:nStart) {
    optimWeights <- optim(weights[i, 1:nPred-1],
                          weightedStack,
                          method = "Nelder-Mead",
                          control = list(maxit = 5000),
                          trainPred = trainPred,
                          trainOutcome = trainOutcome)
    
    weights[i, 'SSE'] <- optimWeights$value
    weights[i, 1:(nPred - 1)] <- optimWeights$par
    weights[i, nPred] <- 1 - sum(weights[i, 1:nPred - 1])
  }
  
  # output the set of weights that give the minimum SSE
  weights <- weights[order(weights$SSE), ]
  as.vector(unlist(weights[1, 1:nPred]))
}

# run the optimisation
set.seed(1234)
weights <- optimWeights(valPred, valOutcome)
weights

weightedSum <- rowSums(vapply(seq(1:20),
                              function(x) {testPred[, x] * weights[x] },
                              numeric(148))
                       )

constrainStackVal  <- data.frame(predict = weightedSum, 
                                 actual = testOutcome)

# not as good as elastic net or averaging, better than
# level one data and simple linear stacked regression
caret::RMSE(constrainStackVal$predict, constrainStackVal$actual)
caret::R2(constrainStackVal$predict, constrainStackVal$actual)

#################################################
# feture weighted linear stacking

# select superplasticiser (zero or non-zero) as metafeature
# (note- this was decided by some trial and error)
# choosing meta features is a bit of a 'black art', ideally
# want to identify important regions in predictive space

# for 20 models, will have 20x2 = 40 predictive inputs now

# data frame of metafeatures
metas <- data.frame(meta0 = 1,
                    meta1 = ifelse(valInput$Superplasticizer == 0, 1, 0))

# generate new data by combining predictive inputs with metafeatures
fwInput <- function(metaFeatures, modelPredictions) {
  out <- matrix(NA, nrow = nrow(modelPredictions), ncol = (ncol(modelPredictions) * ncol(metas)))
  count <- 1
  for (i in 1:ncol(metas)) {
    for (j in 1:ncol(modelPredictions)) {
      out[, count] <- metas[, i] * modelPredictions[, j]
      count <- count + 1
    }
  }
  as.data.frame(out)
}

fwInputs <- fwInput(metas, valPred)

# Perform regularised regression with enet
set.seed(1234)
enetStackFW <- train(x = fwInputs,
                     y = valOutcome,
                     method = "enet",
                     preProcess = c("center", "scale"),
                     tuneGrid = enetGrid,
                     trControl = ctrl)

# evaluate on test set
metasT <- data.frame(meta0 = 1,
                     meta1 = ifelse(testInput$Superplasticizer == 0, 1, 0))

fwInputsT <- fwInput(metasT, testPred)
predict(enetStackFW, fwInputsT)

enetStackFWLS <- data.frame(predict = predict(enetStackFW, fwInputsT),
                           actual = testOutcome)

# seems to improve (slightly) on elastic net without metafeatures
caret::RMSE(enetStackFWLS$predict, enetStackFWLS$actual)
caret::R2(enetStackFWLS$predict, enetStackFWLS$actual)

# residuals fwls
residDataFW <- testInput
residDataFW$predict <- enetStackFWLS$predict
residDataFW$actual <- testOutcome
residDataFW$resid <- residDataFW$predict - residDataFW$actual

ggplot(residDataFW, aes(predict, resid)) +
  geom_point(alpha = 0.5, cex = 2) + 
  theme_bw() + 
  geom_abline(intercept = 0, slope = 0, alpha = 0.7, linetype = "dashed") +
  #  scale_color_manual(values = philTheme()) +
  xlab("Predicted Compressive Strength") + 
  ylab("Residual") + 
  theme(text = element_text(size = 18))

ggplot(residDataFW, aes(predict, actual)) +
  geom_point(alpha = 0.5, cex = 3) + 
  theme_bw() + 
  geom_abline(intercept = 0, slope = 1, alpha = 0.7, linetype = "dashed") +
  #  scale_color_manual(values = philTheme()) +
  xlab("Predicted Compressive Strength") + 
  ylab("Residual") + 
  theme(text = element_text(size = 18))