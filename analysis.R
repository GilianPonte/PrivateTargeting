rm(list = ls(all=TRUE))
library(ggplot2)
library(dplyr)
library(knitr)
library(patchwork)
library(ggrepel)
library(tidyverse)
library(glmnet)
library(ggridges)
require(gridExtra)
library(rpart)
library(summarytools)
library(ggpubr)
library(randomForest)
library(ggh4x)
set.seed(1)
options(scipen = 999)

# simulation study --------------------------------------------------------
#install_github("xnie/rlearner")
library(rlearner)
data_simulation = function(n) {
  x = stats::model.matrix(~.-1, data.frame("covariate_1" = rnorm(n), "covariate_2"= rnorm(n), "covariate_3" = rnorm(n), "covariate_4" = rnorm(n), "covariate_5" = rnorm(n), "covariate_6" = rnorm(n)))
  p = 0.5
  w = as.numeric(rbinom(n,1,p)==1)
  m = pmax(0, x[,1] + x[,2], x[,3]) + pmax(0, x[,4] + x[,5])
  tau = x[,1] + log(1 + exp(x[,2]))^2
  mu1 = m + tau/2
  mu0 = m - tau/2
  y = w*mu1 + (1-w) * mu0 + 0.5*rnorm(n)
  list(x=x, w=w, y=y, p=p, m=m, mu0=mu0, mu1=mu1, tau=tau)
}

set.seed(1)
data = data_simulation(1000)
data$x
write.csv(data$x, "x_square.csv", row.names = F)
data$w
write.csv(data$w, "w_square.csv", row.names = F)
data$y
write.csv(data$y, "y_square.csv", row.names = F)
mean(data$tau) # true tau average 0.9299489
write.csv(data$tau, "tau.csv", row.names = F)
plot(density(data$tau))
mean(data$y)

# without privacy ---------------------------------------------------------
## rboost
ate_collect = c()
for (i in 1:100){
  print(i)
  set.seed(i)
  fit <- rlearner::rboost(x = data$x, w = data$w, y = data$y, k_folds = 5, nthread=1, verbose = TRUE)
  ate = predict(fit, data$x)
  ate_collect = rbind(ate_collect, mean(ate))
}
ate_collect

rboost = read.csv("rboost.csv")
mean(rboost$V1)

# obtain CATE
set.seed(1)
fit <- rlearner::rboost(x = data$x, w = data$w, y = data$y, k_folds = 5, nthread=1, verbose = TRUE)
cate_boost = predict(fit, data$x)
hist(cate_boost)

## rlasso
ate_collect = c()
for (i in 1:100){
  print(i)
  set.seed(i)
  fit <- rlearner::rlasso(x = data$x, w = data$w, y = data$y, k_folds = 5)
  ate = predict(fit, data$x)
  ate_collect = rbind(ate_collect, mean(ate))
}

mean(ate_collect)
sd(ate_collect)
rlasso = ate_collect
rlasso = read.csv("rlasso.csv")
mean(rlasso$V1)

# obtain CATE
set.seed(1)
fit <- rlearner::rlasso(x = data$x, w = data$w, y = data$y)
cate_rlasso = predict(fit, data$x)
hist(cate_rlasso)

## causal forest
ate_collect_causal_forest = c()
for (i in 1:100){
  print(i)
  set.seed(i)
  causalforest <- grf::causal_forest(
    X = data$x,
    Y = data$y,
    W = data$w,
    num.trees = 1000
  )
  # cate from random forest
  CATE_causalforest_sim1 <- predict(causalforest, newdata = data$x, type = "vector")
  print(mean(CATE_causalforest_sim1$predictions))
  ate_collect_causal_forest = rbind(ate_collect_causal_forest, mean(CATE_causalforest_sim1$predictions))
}
ate_collect_causal_forest = read.csv("ate_collect_causal_forest.csv")
mean(ate_collect_causal_forest$V1)

# obtain CATE
set.seed(1)
causalforest <- grf::causal_forest(
  X = data$x,
  Y = data$y,
  W = data$w,
  num.trees = 1000
)
# obtain cate from random forest
cate_causalforest <- predict(causalforest, newdata = data$x, type = "vector")
hist(cate_causalforest$predictions)

## causal network results
ate_collect_nn_logit = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square.csv", header = F)
causal_network_logit = ate_collect_nn_logit$V1
hist(causal_network_logit)
mean(causal_network_logit)
sd(causal_network_logit)

# CATE from causalnetwork
cate_causalnetwork_sim1 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square.csv", header = F)

# ate
boxplots = rbind(data.frame(CATE = rlasso$V1, method = "Rlasso"),data.frame(CATE = rboost$V1, method = "Rboost"), data.frame(CATE = ate_collect_causal_forest$V1,method = "Causal forest"), data.frame(CATE = causal_network_logit,method = "CNN"))
boxplots$method = factor(boxplots$method, levels = c("Rlasso","Rboost","Causal forest", "CNN"))
means <- aggregate(CATE ~  method, boxplots, mean)
boxplots %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot()  + theme_minimal() + geom_hline(yintercept = mean(data$tau), col = 'red') + theme_bw(base_size = 14) + theme(legend.position="bottom") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 14, color = 'black'),legend.text=element_text(size=14, color = 'black')) + ylab("E[tau(X)]")+ xlab("")

# cate
cate_all = rbind(data.frame(CATE = data$tau, method = "true"), data.frame(CATE = cate_rlasso, method = "Rlasso"),data.frame(CATE = cate_boost, method = "Rboost"), data.frame(CATE = cate_causalforest$predictions, method = "Causal forest"), data.frame(CATE = cate_causalnetwork_sim1$V1, method = "CNN"))
cate_all$method = factor(cate_all$method, levels = c("Rlasso","Rboost","Causal forest", "CNN", "true"))
means <- aggregate(CATE ~  method, boxplots, mean)

cate_all %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() + theme_minimal() + theme_bw(base_size = 14) + theme(legend.position="none") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 14, color = 'black'),legend.text=element_text(size=14, color = 'black'))

require(gridExtra)
cate_plot = cate_all %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() + theme_minimal() + theme_bw(base_size = 13) + theme(legend.position="none") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),legend.text=element_text(size=14, color = 'black')) + ylab("tau(X)") + xlab("") 
ate_plot = boxplots %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot()  + theme_minimal() + geom_hline(yintercept = mean(data$tau), col = 'red') + theme_bw(base_size = 13) + theme(legend.position="bottom") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),legend.text=element_text(size=14, color = 'black')) + ylab("E[tau(X)]")+ xlab("")
grid.arrange(ate_plot, cate_plot, nrow = 1)
cate_all$estimand = "estimated tau(X)"
boxplots$estimand = "estimated E[tau(X)]"
simulation1_ = rbind(boxplots, cate_all)

library(geomtextpath)
simulation1_ %>% ggplot(aes(x = CATE, y = method, group = as.factor(method))) + geom_violin(width=1, alpha=.5) + 
  geom_boxplot(width=.1, cex=.2) + theme_minimal() + 
  geom_vline(data=filter(simulation1_, estimand == "estimated E[tau(X)]"), aes(xintercept = 0.92), col = 'red')  +
  facet_grid2(~estimand, scales = "free", independent = TRUE) + 
  theme_bw(base_size = 13) + theme(legend.position="bottom") + 
  theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), 
        strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),
        legend.text=element_text(size=14, color = 'black')) + ylab("")+ xlab("") + coord_flip()


# sorting
uplift_gather = c()
sorting = cbind(data.frame(rlasso = cate_rlasso),data.frame(rboost = cate_boost), data.frame(causal_forest= cate_causalforest$predictions), data.frame(CNN = cate_causalnetwork$V1), data.frame(true = data$tau))
sorting1 = sorting %>% pivot_longer(cols = c(true,rlasso,rboost,causal_forest,CNN))
sorting_selection = data.frame(customer = 1:nrow(sorting))
for (percent in c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,0.2)){
  print(percent)
  top = floor(1000 * percent)
  sorting_selection$selection_tau = 0
  sorting_selection$selection_true = 0
  sorting_selection$selection_rboost = 0
  sorting_selection$selection_rlasso = 0
  sorting_selection$selection_causal_forest = 0
  sorting_selection$selection_random = 0
  sorting_selection$cost = 0.5
  sorting_selection$true = sorting$true
  sorting_selection$selection_random = sample(x = c(0,1), size = nrow(sorting), replace = TRUE, prob= c(1-percent,percent))
  sorting_selection$selection_true[as.data.frame(sort(sorting$true, 
                                                      decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  sorting_selection$selection_tau[as.data.frame(sort(sorting$CNN, 
                                                     decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  sorting_selection$selection_rboost[as.data.frame(sort(sorting$rboost, 
                                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  sorting_selection$selection_rlasso[as.data.frame(sort(sorting$rlasso, 
                                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  sorting_selection$selection_causal_forest[as.data.frame(sort(sorting$causal_forest, 
                                                               decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  overlap = sorting_selection %>% dplyr::select(customer, selection_true, selection_tau, selection_rboost, selection_rlasso,
                                                selection_causal_forest, selection_random) %>% 
    summarize(overlap_CNN = table(selection_true, selection_tau)[2,2]/sum(selection_true),
              overlap_rboost = table(selection_true, selection_rboost)[2,2]/sum(selection_true),
              overlap_rlasso = table(selection_true, selection_rlasso)[2,2]/sum(selection_true),
              overlap_causal_forest = table(selection_true, selection_causal_forest)[2,2]/sum(selection_true),
              overlap_random = table(selection_true, selection_random)[2,2]/sum(selection_true))
  
  uplift = sorting_selection %>% dplyr::select(true,cost, selection_true, selection_tau, selection_rboost, selection_rlasso,
                                               selection_causal_forest, selection_random) %>% 
    pivot_longer(c(selection_true, selection_tau,selection_rboost, selection_rlasso,
                   selection_causal_forest, selection_random)) %>% 
    group_by(name) %>% summarize(profit = (sum(true*value) - sum(cost*value)))
  
  results = cbind(uplift, random, percent) 
  uplift = as.data.frame(results)
  uplift_gather = rbind(uplift_gather, uplift)
}

uplift_gather$name[uplift_gather$name == "selection_tau"] = "CNN"
uplift_gather$name[uplift_gather$name == "selection_rlasso"] = "rlasso"
uplift_gather$name[uplift_gather$name == "selection_rboost"] = "rboost"
uplift_gather$name[uplift_gather$name == "selection_causal_forest"] = "causal forest"
uplift_gather$name[uplift_gather$name == "selection_random"] = "random"
uplift_gather$name[uplift_gather$name == "selection_true"] = "true"

uplift_gather %>% filter(name != "random") %>%
  ggplot(aes(x = percent*100, y = profit, color = name)) + geom_point(size = 2.5) + geom_line() + theme_bw() +
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_gather$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "red", "blue", "green", "grey")) 

# sim 1: first privacy protection -----------------------------------------
causal_network_logit = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square.csv", header = F)$V1
eps_sim001 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_001.csv", header = F)
eps_sim005 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_005.csv", header = F)
eps_sim05 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_05.csv", header = F)
eps_sim1 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_1.csv", header = F)
eps_sim3 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_3.csv", header = F)
eps_sim13 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_13.csv", header = F)
eps_sim50 = read.csv("ATE_sim1_epochs_100_batch_100_folds_5_square_50.csv", header = F)

epsilons_sim_1 = c(0.01, 0.05, 0.5, 1, 3, 13, 50)
eps_sim1 = data.frame(CATE_001 = eps_sim001$V1, CATE_005 = eps_sim005$V1, CATE_05 = eps_sim05$V1, CATE_1 = eps_sim1$V1, CATE_3 = eps_sim3$V1, CATE_13 = eps_sim13$V1, CATE_50 = eps_sim50$V1, causal_nn = causal_network_logit) %>% pivot_longer(cols = c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13, CATE_50, causal_nn))
colnames(eps_sim1) = c("eps", "CATE")
eps_sim1 %>%
  group_by(eps) %>%
  summarise(CATE_mean = mean(CATE),
            CATE_sd = sd(CATE),
            n = n())

eps_sim1$eps[eps_sim1$eps == "CATE_001"] = "0.01"
eps_sim1$eps[eps_sim1$eps == "CATE_005"] = "0.05"
eps_sim1$eps[eps_sim1$eps == "CATE_05"] = "0.5"
eps_sim1$eps[eps_sim1$eps == "CATE_1"] = "1"
eps_sim1$eps[eps_sim1$eps == "CATE_3"] = "3"
eps_sim1$eps[eps_sim1$eps == "CATE_13"] = "13"
eps_sim1$eps[eps_sim1$eps == "CATE_50"] = "50"
eps_sim1$eps[eps_sim1$eps == "causal_nn"] = "CNN"

means <- aggregate(CATE ~  eps, eps_sim1, mean)
eps_sim1$eps = factor(eps_sim1$eps, levels = c("0.01", "0.05", "0.5", "1", "3", "13", "50", "CNN"))

eps_sim1 %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_minimal(base_size = 13) + geom_hline(yintercept = mean(data$tau), col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("E[tau(X)]") + xlab("epsilon")

## CATE with epsilons
cate_causalnetwork_sim1 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square.csv", header = F)
CATE_eps_sim001 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_001.csv", header = F)
CATE_eps_sim005 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_005.csv", header = F)
CATE_eps_sim05 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_05.csv", header = F)
CATE_eps_sim1 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_1.csv", header = F)
CATE_eps_sim3 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_3.csv", header = F)
CATE_eps_sim13 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_13.csv", header = F)
CATE_eps_sim50 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square_50.csv", header = F)

CATE_eps_sim1 = data.frame(CATE_001 = CATE_eps_sim001$V1, CATE_005 = CATE_eps_sim005$V1, CATE_05 = CATE_eps_sim05$V1, CATE_1 = CATE_eps_sim1$V1, CATE_3 = CATE_eps_sim3$V1, CATE_13 = CATE_eps_sim13$V1,
                           CATE_50 = CATE_eps_sim50$V1, CATE = cate_causalnetwork_sim1$V1,real = data$tau, customer= 1:1000) 
CATE_eps_sim1_longer = CATE_eps_sim1 %>% pivot_longer(cols = c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13,CATE_50,CATE, real))
colnames(CATE_eps_sim1_longer) = c("customer","eps", "CATE")
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_001"] = "0.01"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_005"] = "0.05"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_05"] = "0.5"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_1"] = "1"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_3"] = "3"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_13"] = "13"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE_50"] = "50"
CATE_eps_sim1_longer$eps[CATE_eps_sim1_longer$eps == "CATE"] = "CNN"

CATE_eps_sim1_longer$eps = factor(CATE_eps_sim1_longer$eps, levels = c("0.01", "0.05", "0.5", "1", "3", "13", "50","CNN", "real"))
CATE_eps_sim1_longer %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + 
  theme_minimal(base_size = 13) + scale_color_grey(start = 0.0, end = 0.8) + 
  geom_hline(yintercept = mean(data$tau), col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("tau(X)") + xlab("epsilon")
ATE_priv = eps_sim1 %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + geom_hline(yintercept = 0.9299489, col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("estimated E[tau(X)]") + xlab("privacy risk (epsilon)")
CATE_priv= CATE_eps_sim1_longer %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + scale_color_grey(start = 0.0, end = 0.8) + geom_hline(yintercept = 0.9299489, col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("estimated tau(X)") + xlab("privacy risk (epsilon)")
library(gridExtra)
grid.arrange(ATE_priv, CATE_priv, nrow = 1)

# sorting
uplift_gather = c()
for (percent in c(0.01,0.015,0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,0.075,0.1,0.15,0.175,0.2, 0.225,0.25, 0.275,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)){
  print(percent)
  results = CATE_eps_sim1 %>% mutate(real_est = real) %>% pivot_longer(c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13,CATE_50, CATE, real)) %>%  
    group_by(name) %>% 
    top_n(value, n = floor(1000 * percent)) %>% 
    slice_head(n = floor(1000* percent)) %>% 
    summarize(profit = sum(real_est))
  random_est = CATE_eps_sim1 %>% select(real = c(real)) %>% 
    sample_n(floor(1000 * percent)) %>% summarize(profit = sum(real))
  
  results = rbind(results,c(name = "random",random_est))
  uplift = as.data.frame(results)
  uplift$percentage = percent
  uplift_gather = rbind(uplift_gather, uplift)
}

uplift_gather$name[uplift_gather$name == "CATE_001"] = "0.01"
uplift_gather$name[uplift_gather$name == "CATE_005"] = "0.05"
uplift_gather$name[uplift_gather$name == "CATE_05"] = "0.5"
uplift_gather$name[uplift_gather$name == "CATE_1"] = "1"
uplift_gather$name[uplift_gather$name == "CATE_3"] = "3"
uplift_gather$name[uplift_gather$name == "CATE_13"] = "13"
uplift_gather$name[uplift_gather$name == "CATE_50"] = "50"
uplift_gather$name[uplift_gather$name == "CATE"] = "CNN"

uplift_gather %>% ggplot(aes(x = percentage, y = profit, color = name, shape = name)) + geom_point(size = 2.5) +geom_line() + theme_minimal() + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13)) +
  scale_shape_manual(values = 1:length(unique(uplift_gather$name)))

sim1_sorting = uplift_gather %>% ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + annotate(
    'text',
    x = 25,
    y = 1100,
    label = 'CNN (black) captures \n the true targeting policy accurately (green).', 
    size = 4
  ) + annotate(
    'curve',
    x = 25, # Play around with the coordinates until you're satisfied
    y = 1000,
    yend = 870,
    xend = 25,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'text',
    x = 70,
    y = 300,
    label = 'Other targeting policies are \n approximately random.', 
    size = 4
  ) + annotate(
    'curve',
    x = 70, # Play around with the coordinates until you're satisfied
    y = 350,
    yend = 550,
    xend = 65,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
                                   axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_gather$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,100,200,300,400,500,600,700,800,900,1000,1100,1200)) +
  theme(legend.position="bottom") + 
  scale_color_manual(values = c("orange", "purple", "#ddb321", "blue",  "grey", "#9ea900", "#954010","black", "red", "darkgreen")) 

sim1_sorting

# sim 1: second privacy strategy --------------------------------------------------
cate_causalnetwork_sim1 = read.csv("CATE_sim1_epochs_100_batch_100_folds_5_square.csv", header = F)
second = data.frame(tau = data$tau, causal_neural_tau = cate_causalnetwork_sim1$V1)
results_in_sim1 = c()
results_in_sim1_uplift = c()
results_in_sim1_revenue = c() 


dp_clipping = function(CATE, min_CATE, max_CATE, epsilon){
  set.seed(1)
  clip <- function(x, a, b) {
    ifelse(x < a, a, ifelse(x > b, b, x))
  }
  clipped = clip(CATE, a =min_CATE, b = max_CATE)
  sensitivity = max_CATE - min_CATE
  clipped = DPpack::LaplaceMechanism(clipped, epsilon, sensitivity)
  return(clipped)
}

percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  top = floor(1000 * percent)
  second$customer = 1:1000
  second$selection_tau = 0
  second$selection_true = 0
  second$selection_tau_3 = 0
  second$selection_tau_1 = 0
  second$selection_tau_50 = 0
  second$selection_tau_05 = 0
  second$selection_tau_005 = 0
  second$selection_tau_001 = 0
  second$selection_tau_5 = 0
  second$random = sample(x = c(0,1), size = 1000, replace = TRUE, prob= c(1-percent,percent))
  second$cost = 0.5
  second$selection_true[as.data.frame(sort(data$tau, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  # now with local dp
  min_CATE = 0
  max_CATE = 1
  clipped_001 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.01)
  clipped_005 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
  clipped_05 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
  clipped_1 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
  clipped_3 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
  clipped_5 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)
  clipped_50 = dp_clipping(second$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 50)
  
  second$selection_tau[as.data.frame(sort(second$causal_neural_tau, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  second$selection_tau_001[as.data.frame(sort(clipped_001, 
                                              decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_005[as.data.frame(sort(clipped_005, 
                                            decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_05[as.data.frame(sort(clipped_05, 
                                           decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_1[as.data.frame(sort(clipped_1, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_3[as.data.frame(sort(clipped_3, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_5[as.data.frame(sort(clipped_5, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second$selection_tau_50[as.data.frame(sort(clipped_50, 
                                            decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  
  overlap = second %>% dplyr::select(customer, selection_true, selection_tau,selection_tau_001, selection_tau_005, 
                                     selection_tau_05,
                                   selection_tau_1,selection_tau_3,selection_tau_5,selection_tau_50, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_001 = table(selection_true, selection_tau_001)[2,2]/sum(selection_true),
              overlap_05 = table(selection_true, selection_tau_05)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, selection_tau_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, selection_tau_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, selection_tau_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, selection_tau_5)[2,2]/sum(selection_true),
              overlap_50 = table(selection_true, selection_tau_50)[2,2]/sum(selection_true))
  
  uplift = second %>% dplyr::select(tau,cost, selection_true, selection_tau, selection_tau_001, selection_tau_005, 
                                  selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5,selection_tau_50, 
                                  random) %>% 
    pivot_longer(c(selection_true, selection_tau,selection_tau_001, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5,selection_tau_50, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(tau*value) - sum(cost*value)))
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_sim1 = rbind(results_in_sim1, overlap)
  results_in_sim1_uplift = rbind(results_in_sim1_uplift, uplift)
}
results_in_sim1
results_in_sim1_uplift

results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau"] = "CNN"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_001"] = "0.01"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_005"] = "0.05"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_05"] = "0.5"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_1"] = "1"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_3"] = "3"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_5"] = "5"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_50"] = "50"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_true"] = "real"

sim1_second = results_in_sim1_uplift %>% filter(name != "CNN") %>% filter(name != "50") %>% 
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() +
  theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_gather$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,100,200,300,400,500,600,700,800,900,1000,1100,1200)) +
  theme(legend.position="bottom") + 
  scale_color_manual(values = c("orange", "purple", "#ddb321", "blue",  "grey", "#9ea900", "red", "darkgreen")) + annotate(
    'text',
    x = 15,
    y = 800,
    label = 'e = 5 now close to \n the true targeting policy',
    size = 4
  ) + annotate(
    'curve',
    x = 10, # Play around with the coordinates until you're satisfied
    y = 750,
    yend = 550,
    xend = 30,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'text',
    x = 80,
    y = 100,
    label = 'Privacy protected policies now \n outperform random targeting policy.',
    size = 4
  ) + annotate(
    'curve',
    x = 80, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 400,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 80, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 320,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 80, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 475,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) 
sim1_second

# sim 1: third privacy strategy ------------------------------------------------------------------
first = data.frame(tau = data$tau, causal_neural_tau = cate_causalnetwork_sim1$V1)
results_in_sim1 = c()
results_in_sim1_uplift = c()
results_in_sim1_revenue = c() 
percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  top = floor(1000 * percent)
  first$customer = 1:1000
  first$selection_tau = 0
  first$selection_true = 0
  first$selection_tau_3 = 0
  first$selection_tau_1 = 0
  first$selection_tau_05 = 0
  first$selection_tau_005 = 0
  first$selection_tau_5 = 0
  first$random = sample(x = c(0,1), size = 1000, replace = TRUE, prob= c(1-percent,percent))
  first$cost = 0.5
  first$selection_true[as.data.frame(sort(first$tau, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  first$selection_tau[as.data.frame(sort(first$causal_neural_tau, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = first$selection_tau
  epsilon_range = c(0.05,0.5,1,3,5)
  for (epsilon in epsilon_range){
    print(epsilon)
    P = matrix(nrow = 2, ncol = 2)
    diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
    P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
    
    responses = c()
    for (i in 1:length(pop)){
      #print(i)
      if(pop[i] == 0){responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob= P[1,]))}
      else{responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob=P[2,]))}
    }
    if(epsilon == 0.5){
      first$selection_tau_05 = responses 
      index05_0 = which(first$selection_tau_05 == 0)
      index05 = which(first$selection_tau_05 == 1)
      first$selection_tau_05 = 0
      if(top > length(index05)){
        first$selection_tau_05[sample(index05, length(index05))] = 1
        first$selection_tau_05[sample(index05_0, top - length(index05))] = 1
      }else{
        first$selection_tau_05[sample(index05, top)] = 1
      }
    } else if(epsilon == 0.05){
      first$selection_tau_005 = responses 
      index005_0 = which(first$selection_tau_005 == 0)
      index005 = which(first$selection_tau_005 == 1)
      first$selection_tau_005 = 0
      if(top > length(index005)){
        first$selection_tau_005[sample(index005, length(index005))] = 1
        first$selection_tau_005[sample(index005_0, top - length(index005))] = 1
      }else{
        first$selection_tau_005[sample(index005, top)] = 1
      }
    } else if(epsilon == 5){
      first$selection_tau_5 = responses 
      index5_0 = which(first$selection_tau_5 == 0)
      index5 = which(first$selection_tau_5 == 1)
      first$selection_tau_5 = 0
      if(top > length(index5)){
        first$selection_tau_5[sample(index5, length(index5))] = 1
        first$selection_tau_5[sample(index5_0, top - length(index5))] = 1
      }else{
        first$selection_tau_5[sample(index5, top)] = 1
      }
    } else if(epsilon == 1){
      first$selection_tau_1 = responses 
      index1_0 = which(first$selection_tau_1 == 0)
      index1 = which(first$selection_tau_1 == 1)
      first$selection_tau_1 = 0
      if(top > length(index1)){
        first$selection_tau_1[sample(index1, length(index1))] = 1
        first$selection_tau_1[sample(index1_0, top - length(index1))] = 1
      }else{
        first$selection_tau_1[sample(index1, top)] = 1
      }
    } else {
      first$selection_tau_3 = responses 
      index3_0 = which(first$selection_tau_3 == 0)
      index3 = which(first$selection_tau_3 == 1)
      first$selection_tau_3 = 0
      if(top > length(index3)){
        first$selection_tau_3[sample(index3, length(index3))] = 1
        first$selection_tau_3[sample(index3_0, top - length(index3))] = 1
      }else{
        first$selection_tau_3[sample(index3, top)] = 1
      }
    }
  }
  overlap = first %>% dplyr::select(customer, selection_true, selection_tau, selection_tau_005, selection_tau_05,
                            selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_true, selection_tau_05)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, selection_tau_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, selection_tau_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, selection_tau_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, selection_tau_5)[2,2]/sum(selection_true))
  uplift = first %>% dplyr::select(tau,cost, selection_true, selection_tau, selection_tau_005, 
                              selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                              random) %>% 
    pivot_longer(c(selection_true, selection_tau, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(tau*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_sim1 = rbind(results_in_sim1, overlap)
  results_in_sim1_uplift = rbind(results_in_sim1_uplift, uplift)
}
results_in_sim1
results_in_sim1_uplift

results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau"] = "CNN"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_005"] = "0.05"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_05"] = "0.5"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_1"] = "1"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_3"] = "3"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_tau_5"] = "5"
results_in_sim1_uplift$name[results_in_sim1_uplift$name == "selection_true"] = "real"

uplift_plot_sim1 = results_in_sim1_uplift %>% filter(name != "CNN") %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + annotate(
    'text',
    x = 20,
    y = 820,
    label = 'Privacy protected policies are closer \n to true policy.', 
    size = 4
  ) + annotate(
    'curve',
    x = 20, # Play around with the coordinates until you're satisfied
    y = 810,
    yend = 700,
    xend = 30,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + theme_bw() +
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
                                   axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(results_in_sim1_uplift$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "#ddb321", "blue",  "grey", "#9ea900", "red", "darkgreen"))

uplift_plot_sim1 + annotate(
  'text',
  x = 70,
  y = 100,
  label = 'The stronger the privacy protection, \n the closer we get to a random sampling targeting policy.',
  size = 4) + annotate(
    'curve',
    x = 85, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 490,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 85, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 690,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 85, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 760,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 85, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 340,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 85, # Play around with the coordinates until you're satisfied
    y = 150,
    yend = 310,
    xend = 70,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) 


# second simulation -------------------------------------------------------
set.seed(1)
setup_c_adapt = function(n) {
  x = stats::model.matrix(~.-1, data.frame("covariate_1" = rnorm(n), "covariate_2"= rnorm(n), "covariate_3" = rbinom(n, 10, prob = 0.25), "covariate_4" = rbinom(n, 1, prob = 0.95), "covariate_5" = rnorm(n), "covariate_6" = rlnorm(n, meanlog = 0, sdlog = 1), "covariate_7" = rlnorm(n, meanlog = 1, sdlog = 0.1), "covariate_8" = rexp(n, rate = 1), "covariate_9" = rnorm(n), "covariate_10" = rexp(n, rate = 1),"covariate_11" = rnorm(n), "covariate_12"= rnorm(n), "covariate_13" = rbinom(n, 10, prob = 0.25), "covariate_14" = rbinom(n, 1, prob = 0.95), "covariate_15" = rnorm(n), "covariate_16" = rlnorm(n, meanlog = 0, sdlog = 1), "covariate_17" = rlnorm(n, meanlog = 1, sdlog = 0.1), "covariate_18" = rexp(n, rate = 1), "covariate_19" = rnorm(n), "covariate_20" = rexp(n, rate = 1), "covariate_21" = rnorm(n), "covariate_22"= rnorm(n), "covariate_23" = rbinom(n, 10, prob = 0.25), "covariate_24" = rbinom(n, 1, prob = 0.95), "covariate_25" = rnorm(n), "covariate_26" = rlnorm(n, meanlog = 0, sdlog = 1), "covariate_27" = rlnorm(n, meanlog = 1, sdlog = 0.1), "covariate_28" = rexp(n, rate = 1), "covariate_29" = rnorm(n), "covariate_30" = rexp(n, rate = 1)))
  m = 2 * log(1 + exp(x[,1] + x[,2] + x[,3]))
  p = 1/(1 + exp(x[,2] + x[,3] + x[,10]))
  w = as.numeric(rbinom(n,1,p)==1)
  tau = log(1 + exp(x[,2] + x[,3] + x[,12] + x[,25]))**2
  mu1 = m + tau/2
  mu0 = m - tau/2
  y = w*mu1 + (1-w) * mu0 + 0.5*rnorm(n)
  list(x=x, w=w, y=y, p=p, m=m, mu0=mu0, mu1=mu1, tau=tau)
}


data2 = setup_c_adapt(100000)
mean(data2$tau)
sd(data2$tau)
hist(data2$tau)
mean(data2$y)

#write.csv(data2$x, "x_toy.csv", row.names = F)
#write.csv(data2$w, "w_toy.csv", row.names = F)
#write.csv(data2$y, "y_toy.csv", row.names = F)
#write.csv(data2$tau, "tau_toy.csv") # true tau = 2.786143

mean(data2$tau) # true tau = 11.12626
hist(data2$tau)

## rlasso
ate_collect = c()
for (i in 1:100){
  set.seed(i)
  print(i)
  fit <- rlearner::rlasso(x = data2$x, w = data2$w, y = data2$y, k_folds = 5)
  ate = predict(fit, data2$x)
  ate_collect = rbind(ate_collect, mean(ate))
  print(mean(ate))
}

rlasso = ate_collect
mean(rlasso)

#write.csv(rlasso,"rlasso/rlasso.csv", row.names = F)
rlasso = read.csv("rlasso/rlasso.csv")
mean(rlasso$V1)
sd(rlasso$V1)

# obtain CATE
set.seed(1)
fit <- rlearner::rlasso(x = data2$x, w = data2$w, y = data2$y, k_folds = 5)
cate_rlasso = predict(fit, data2$x)
mean(cate_rlasso)
sd(cate_rlasso)

## rboost
ate_collect = c()
for (i in 1:2){
  set.seed(i)
  print(i)
  fit <- rlearner::rboost(x = data2$x, w = data2$w, y = data2$y, k_folds = 5)
  ate = predict(fit, data2$x)
  print(mean(ate))
  ate_collect = rbind(ate_collect, mean(ate))
}

rboost = ate_collect
mean(rboost)
sd(rboost)

#write.csv(rboost, "rboost/rboost_ate_collect_simulation_2.csv", row.names = F)
ate_collect_rboost = read.csv("rboost/rboost_ate_collect_simulation_2.csv")
hist(ate_collect_rboost$V1)
mean(ate_collect_rboost$V1)
sd(ate_collect_rboost$V1)

## causal forest
rm(ate)
rm(fit)
ate_collect_causal_forest = c()
for (i in 91:100){
  set.seed(i)
  print(i)
  causalforest <- grf::causal_forest(
    X = data2$x,
    Y = data2$y,
    W = data2$w,
    num.trees = 1000
  )
  #print(mean(data2$y))
  # cate from random forest
  CATE_causalforest_sim2 <- predict(causalforest, newdata = data2$x, type = "vector")
  ate_collect_causal_forest = rbind(ate_collect_causal_forest, mean(CATE_causalforest_sim2$predictions))
  print(mean(CATE_causalforest_sim2$predictions))
}
mean(ate_collect_causal_forest)
#write.csv(ate_collect_causal_forest_sim2, "causal forest/ate_collect_causal_forest_10.csv", row.names = F) 
ate_collect_causal_forest_sim2 = read.csv("causal forest/ate_collect_causal_forest_10.csv")
mean(ate_collect_causal_forest_sim2$V1)
hist(ate_collect_causal_forest_sim2$V1)

# obtain CATE
set.seed(1)
causalforest <- grf::causal_forest(
  X = data2$x,
  Y = data2$y,
  W = data2$w,
  num.trees = 1000
)
# obtain cate from random forest
cate_causalforest <- predict(causalforest, newdata = data2$x, type = "vector")
hist(cate_causalforest$predictions)

## causal network
causal_net_sim2 = read.csv("causal network/ATE_sim2_epochs_100_batch_100_folds_5.csv", header = F)
hist(causal_net_sim2$V1)
mean(causal_net_sim2$V1)
sd(causal_net_sim2$V1)

cate_causalnetwork = read.csv("causal network/CATE_sim2_epochs_100_batch_100_folds_5.csv", header = F)

boxplots = rbind(data.frame(CATE = rlasso$V1, method = "Rlasso"), 
                 data.frame(CATE = ate_collect_causal_forest_sim2$V1 ,method = "Causal forest"), 
                 data.frame(CATE = causal_net_sim2$V1 ,method = "CNN"))
boxplots$method = factor(boxplots$method, levels = c("Rlasso","Rboost","Causal forest", "CNN"))
means <- aggregate(CATE ~  method, boxplots, mean)
boxplots %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot()  + theme_minimal() + geom_hline(yintercept = 11.12626, col = 'red') + theme(axis.text.x = element_text(size = 12),axis.text.y = element_text(size = 12)) + ylab("E[tau(X)]")
# cate
cate_all = bind_rows(data.frame(CATE = data2$tau, method = "true"), 
                 data.frame(CATE = cate_rlasso, method = "Rlasso"), 
                 data.frame(CATE = cate_causalforest$predictions, method = "Causal forest"), 
                 data.frame(CATE = cate_causalnetwork$V1, method = "CNN"))
cate_all$method = factor(cate_all$method, levels = c("Rlasso","Causal forest", "CNN", "true"))
means <- aggregate(CATE ~  method, boxplots, mean)
cate_all %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() + theme_minimal() + 
  theme_bw(base_size = 14) + theme(legend.position="none") + theme(text = element_text(size = 14),
                                                                   axis.text = element_text(size = 14, color = "black"), 
                                                                   strip.text.x = element_text(size = 14), 
                                                                   axis.text.x = element_text(size = 14, color = 'black'),
                                                                   legend.text=element_text(size=14, color = 'black'))

cate_plot = cate_all %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() + theme_minimal() + theme_bw(base_size = 13) + theme(legend.position="none") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),legend.text=element_text(size=14, color = 'black')) + ylab("tau(X)") + xlab("")
ate_plot = boxplots %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot()  + theme_minimal() + geom_hline(yintercept = mean(data2$tau), col = 'red') + theme_bw(base_size = 13) + theme(legend.position="bottom") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),legend.text=element_text(size=14, color = 'black')) + ylab("E[tau(X)]")+ xlab("") + ylim(0,15)
grid.arrange(ate_plot, cate_plot, nrow = 1)

cate_all %>% ggplot(aes(x = CATE, y = method, fill = method)) + geom_density_ridges2() + theme_minimal() + theme_bw(base_size = 13) + theme(legend.position="none") + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),legend.text=element_text(size=14, color = 'black')) + xlab("tau(X)") + xlab("method")

cate_all$estimand = "estimated tau(X)"
boxplots$estimand = "estimated E[tau(X)]"
simulation1_ = rbind(boxplots, cate_all)

simulation1_ %>% ggplot(aes(x = CATE, y = method, group = as.factor(method))) + geom_violin(width=1, alpha=.5) + 
  geom_boxplot(width=.05, cex=.2) + theme_minimal() + 
  geom_vline(data=filter(simulation1_, estimand == "estimated E[tau(X)]"), aes(xintercept = mean(data2$tau)), col = 'red')  +
  facet_grid2(~estimand, scales = "free", independent = TRUE) + 
  theme_bw(base_size = 13) + theme(legend.position="bottom") + 
  theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), 
        strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 12, color = 'black'),
        legend.text=element_text(size=14, color = 'black')) + ylab("")+ xlab("")  + coord_flip()

## simulation with epsilons
eps_sim001 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_001.csv", header = F)
eps_sim005 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_005.csv", header = F)
eps_sim05 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_05.csv", header = F)
eps_sim1_2 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_1.csv", header = F)
eps_sim3 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_3.csv", header = F)
eps_sim13 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_13.csv", header = F)
eps_sim50 = read.csv("epsilon/ATE_sim2_epochs_100_batch_100_folds_5_square_50.csv", header = F)

epsilons_sim_1 = c(0.01, 0.05, 0.5, 1, 3, 13, 50)
eps_sim2 = data.frame(CATE_001 = eps_sim001$V1, CATE_005 = eps_sim005$V1, CATE_05 = eps_sim05$V1, CATE_1 = eps_sim1_2$V1, CATE_3 = eps_sim3$V1, CATE_13 = eps_sim13$V1, CATE_50 = eps_sim50$V1, causal_nn = causal_net_sim2$V1) %>% pivot_longer(cols = c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13,CATE_50, causal_nn))
colnames(eps_sim2) = c("eps", "CATE")
eps_sim2 %>%
  group_by(eps) %>%
  summarise(CATE_mean = mean(CATE),
            CATE_sd = sd(CATE),
            n = n())

eps_sim2$eps[eps_sim2$eps == "CATE_001"] = "0.01"
eps_sim2$eps[eps_sim2$eps == "CATE_005"] = "0.05"
eps_sim2$eps[eps_sim2$eps == "CATE_05"] = "0.5"
eps_sim2$eps[eps_sim2$eps == "CATE_1"] = "1"
eps_sim2$eps[eps_sim2$eps == "CATE_3"] = "3"
eps_sim2$eps[eps_sim2$eps == "CATE_13"] = "13"
eps_sim2$eps[eps_sim2$eps == "CATE_50"] = "50"
eps_sim2$eps[eps_sim2$eps == "causal_nn"] = "CNN"

means <- aggregate(CATE ~  eps, eps_sim2, mean)
eps_sim2$eps = factor(eps_sim2$eps, levels = c("0.01", "0.05", "0.5", "1", "3", "13","50", "CNN"))
eps_sim2 %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_minimal(base_size = 13) + geom_hline(yintercept = 11.12626, col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("E[tau(X)]") + xlab("epsilon")

## CATE with epsilons
cate_causalnetwork = read.csv("causal network/CATE_sim2_epochs_100_batch_100_folds_5.csv", header = F)
CATE_eps_sim001 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_001.csv", header = F)
CATE_eps_sim005 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_005.csv", header = F)
CATE_eps_sim05 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_05.csv", header = F)
CATE_eps_sim1 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_1.csv", header = F)
CATE_eps_sim3 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_3.csv", header = F)
CATE_eps_sim13 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_13.csv", header = F)
CATE_eps_sim50 = read.csv("epsilon/CATE_sim2_epochs_100_batch_100_folds_5_square_50.csv", header = F)

CATE_eps_sim2 = data.frame(CATE_001 = CATE_eps_sim001$V1, CATE_005 = CATE_eps_sim005$V1, CATE_05 = CATE_eps_sim05$V1, CATE_1 = CATE_eps_sim1$V1, CATE_3 = CATE_eps_sim3$V1, CATE_13 = CATE_eps_sim13$V1,CATE_50 = CATE_eps_sim50$V1, CATE = cate_causalnetwork$V1, real = data2$tau, customer= 1:100000) 

CATE_eps_sim2_longer = CATE_eps_sim2 %>% pivot_longer(cols = c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13,CATE_50,CATE, real))
colnames(CATE_eps_sim2_longer) = c("customer","eps", "CATE")
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_001"] = "0.01"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_005"] = "0.05"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_05"] = "0.5"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_1"] = "1"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_3"] = "3"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_13"] = "13"
CATE_eps_sim2_longer$eps[CATE_eps_sim2_longer$eps == "CATE_50"] = "50"

CATE_eps_sim2_longer$eps = factor(CATE_eps_sim2_longer$eps, levels = c("0.01", "0.05", "0.5", "1", "3", "13", "50", "true"))
CATE_eps_sim2_longer %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + scale_color_grey(start = 0.0, end = 0.8) + geom_hline(yintercept = mean(data2$tau), col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("tau(X)") + xlab("epsilon")

ATE_priv_2 = eps_sim2 %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + geom_hline(yintercept = 11.12626, col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("estimated E[tau(X)]") + xlab("privacy risk (epsilon)")
CATE_priv_2= CATE_eps_sim2_longer %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + scale_color_grey(start = 0.0, end = 0.8) + geom_hline(yintercept = 11.12626, col = 'red') + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black")) + ylab("estimated tau(X)") + xlab("privacy risk (epsilon)")

# overall plot
eps_sim1$simulation = "simulation 1"
eps_sim2$simulation = "simulation 2"
eps_sim1$tau = 0.93
eps_sim2$tau = 11.12
eps_sim1$estimand = "estimated E[tau(X)]"
eps_sim2$estimand = "estimated E[tau(X)]"
sim1 = rbind(eps_sim1, eps_sim2)

CATE_eps_sim1_longer$simulation = "simulation 1" 
CATE_eps_sim2_longer$simulation = "simulation 2"
CATE_eps_sim1_longer$tau = 0.93 
CATE_eps_sim2_longer$tau = 11.12
CATE_eps_sim1_longer$estimand = "estimated tau(X)"
CATE_eps_sim2_longer$estimand = "estimated tau(X)"
CATE_eps_sim1_longer$customer = NULL
CATE_eps_sim2_longer$customer = NULL
CATE_eps_sim1_longer$customer_ordered = NULL
sim2 = rbind(CATE_eps_sim2_longer, CATE_eps_sim1_longer)
sims = rbind(sim1, sim2)
sims %>% ggplot(aes(x= as.factor(eps), y= CATE)) + geom_boxplot() + theme_bw(base_size = 13) + geom_hline(aes(yintercept = tau, col = 'red')) + facet_grid2(estimand~simulation, scales = "free", independent = TRUE) + theme(axis.text.x = element_text(size = 13, color = "black"),axis.text.y = element_text(size = 13, color = "black"))+ xlab("privacy risk (epsilon)") + ylab("") + theme(legend.position = "none", strip.text.x = element_text(size = 13))   

# sim 2: first strategy privacy -----------------------------------------
uplift_gather = c()
for (percent in c(0.01,0.015,0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,0.075,0.1,0.15,0.175,0.2, 0.225,0.25, 0.275,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)){
  print(percent)
  results = CATE_eps_sim2 %>% mutate(real_est = real) %>% pivot_longer(c(CATE_001,CATE_005,CATE_05,CATE_1,CATE_3,CATE_13,CATE_50, CATE, real)) %>%  
    group_by(name) %>% 
    top_n(value, n = floor(100000 * percent)) %>% 
    slice_head(n = floor(100000* percent)) %>% 
    summarize(profit = sum(real_est))
  random_est = CATE_eps_sim2 %>% select(real = c(real)) %>% 
    sample_n(floor(100000 * percent)) %>% summarize(profit = sum(real))
  
  results = rbind(results,c(name = "random",random_est))
  uplift = as.data.frame(results)
  uplift$percentage = percent
  uplift_gather = rbind(uplift_gather, uplift)
}

uplift_gather$name[uplift_gather$name == "CATE_001"] = "0.01"
uplift_gather$name[uplift_gather$name == "CATE_005"] = "0.05"
uplift_gather$name[uplift_gather$name == "CATE_05"] = "0.5"
uplift_gather$name[uplift_gather$name == "CATE_1"] = "1"
uplift_gather$name[uplift_gather$name == "CATE_3"] = "3"
uplift_gather$name[uplift_gather$name == "CATE_13"] = "13"
uplift_gather$name[uplift_gather$name == "CATE_50"] = "50"
uplift_gather$name[uplift_gather$name == "CATE"] = "CNN"

uplift_gather %>% ggplot(aes(x = percentage, y = profit, color = name, shape = name)) + geom_point(size = 2.5) +geom_line() + theme_minimal() + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13)) +
  scale_shape_manual(values = 1:length(unique(uplift_gather$name)))

sim2_sorting= uplift_gather %>% ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + annotate(
  'text',
  x = 25,
  y = 1e6,
  label = 'CNN (black) captures \n the true targeting policy accurately (green).', 
  size = 4
) + annotate(
  'curve',
  x = 25, # Play around with the coordinates until you're satisfied
  y = 9e5,
  yend = 800000,
  xend = 25,
  linewidth = 0.5,
  curvature = 0.2,
  arrow = arrow(length = unit(0.2, 'cm'))
) + annotate(
  'text',
  x = 70,
  y = 250000,
  label = 'Other targeting policies are \n approximately random.', 
  size = 4
) + annotate(
  'curve',
  x = 70, # Play around with the coordinates until you're satisfied
  y = 320000,
  yend = 600000,
  xend = 65,
  linewidth = 0.5,
  curvature = 0.2,
  arrow = arrow(length = unit(0.2, 'cm'))
) + theme_bw() + ylab("") +
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_gather$name))) + 
  theme(legend.text=element_text(size=13)) + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6)) +
  theme(legend.position="bottom") + 
  scale_color_manual(values = c("orange", "purple", "#ddb321", "blue",  "grey", "#9ea900", "#954010","black", "red", "darkgreen")) 

ggarrange(sim1_sorting, sim2_sorting, ncol =2, common.legend = TRUE, legend="bottom")


results_in = c()
results_in_uplift = c()
results_in_revenue = c() 
percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  top = floor(100000 * percent)
  data$customer = 1:100000
  data$selection_tau = 0
  data$selection_true = 0
  data$selection_tau_3 = 0
  data$selection_tau_1 = 0
  data$selection_tau_05 = 0
  data$selection_tau_005 = 0
  data$selection_tau_5 = 0
  data$random = sample(x = c(0,1), size = 100000, replace = TRUE, prob= c(1-percent,percent))
  data$cost = 0.5
  data$selection_true[as.data.frame(sort(data2$tau, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau[as.data.frame(sort(data2$causal_neural_tau, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = data$selection_tau
  epsilon_range = c(0.05,0.5,1,3,5)
  for (epsilon in epsilon_range){
    print(epsilon)
    P = matrix(nrow = 2, ncol = 2)
    diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
    P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
    
    responses = c()
    for (i in 1:length(pop)){
      #print(i)
      if(pop[i] == 0){responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob= P[1,]))}
      else{responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob=P[2,]))}
    }
    if(epsilon == 0.5){
      data$selection_tau_05 = responses 
      index05_0 = which(data$selection_tau_05 == 0)
      index05 = which(data$selection_tau_05 == 1)
      data$selection_tau_05 = 0
      if(top > length(index05)){
        data$selection_tau_05[sample(index05, length(index05))] = 1
        data$selection_tau_05[sample(index05_0, top - length(index05))] = 1
      }else{
        data$selection_tau_05[sample(index05, top)] = 1
      }
    } else if(epsilon == 0.05){
      data$selection_tau_005 = responses 
      index005_0 = which(data$selection_tau_005 == 0)
      index005 = which(data$selection_tau_005 == 1)
      data$selection_tau_005 = 0
      if(top > length(index005)){
        data$selection_tau_005[sample(index005, length(index005))] = 1
        data$selection_tau_005[sample(index005_0, top - length(index005))] = 1
      }else{
        data$selection_tau_005[sample(index005, top)] = 1
      }
    } else if(epsilon == 5){
      data$selection_tau_5 = responses 
      index5_0 = which(data$selection_tau_5 == 0)
      index5 = which(data$selection_tau_5 == 1)
      data$selection_tau_5 = 0
      if(top > length(index5)){
        data$selection_tau_5[sample(index5, length(index5))] = 1
        data$selection_tau_5[sample(index5_0, top - length(index5))] = 1
      }else{
        data$selection_tau_5[sample(index5, top)] = 1
      }
    } else if(epsilon == 1){
      data$selection_tau_1 = responses 
      index1_0 = which(data$selection_tau_1 == 0)
      index1 = which(data$selection_tau_1 == 1)
      data$selection_tau_1 = 0
      if(top > length(index1)){
        data$selection_tau_1[sample(index1, length(index1))] = 1
        data$selection_tau_1[sample(index1_0, top - length(index1))] = 1
      }else{
        data$selection_tau_1[sample(index1, top)] = 1
      }
    } else {
      data$selection_tau_3 = responses 
      index3_0 = which(data$selection_tau_3 == 0)
      index3 = which(data$selection_tau_3 == 1)
      data$selection_tau_3 = 0
      if(top > length(index3)){
        data$selection_tau_3[sample(index3, length(index3))] = 1
        data$selection_tau_3[sample(index3_0, top - length(index3))] = 1
      }else{
        data$selection_tau_3[sample(index3, top)] = 1
      }
    }
  }
  overlap = data %>% dplyr::select(customer, selection_true, selection_tau, selection_tau_005, selection_tau_05,
                                   selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_true, selection_tau_05)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, selection_tau_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, selection_tau_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, selection_tau_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, selection_tau_5)[2,2]/sum(selection_true))
  uplift = data %>% dplyr::select(tau,cost, selection_true, selection_tau, selection_tau_005, 
                                  selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                  random, random) %>% 
    pivot_longer(c(selection_true, selection_tau, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(tau*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in = rbind(results_in, overlap)
  results_in_uplift = rbind(results_in_uplift, uplift)
}
results_in
results_in_uplift

results_in_uplift$name[results_in_uplift$name == "selection_tau"] = "CNN"
results_in_uplift$name[results_in_uplift$name == "selection_tau_005"] = "e = 0.05"
results_in_uplift$name[results_in_uplift$name == "selection_tau_05"] = "e = 0.5"
results_in_uplift$name[results_in_uplift$name == "selection_tau_1"] = "e = 1"
results_in_uplift$name[results_in_uplift$name == "selection_tau_3"] = "e = 3"
results_in_uplift$name[results_in_uplift$name == "selection_tau_5"] = "e = 5"
results_in_uplift$name[results_in_uplift$name == "selection_true"] = "true"

uplift_plot_sim2 = results_in_uplift %>% 
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + theme_bw() +
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(results_in$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + theme(legend.position="bottom") +  
  scale_color_manual(values = c("black", "blue",  "grey", "#9ea900", "#954010","#ddb321", "red", "darkgreen")) 


grid.arrange(uplift_plot_sim1, uplift_plot_sim2, nrow = 1)


# sim 2: second privacy strategy -------------------------------------------
cate_causalnetwork = read.csv("causal network/CATE_sim2_epochs_100_batch_100_folds_5.csv", header = F)
second_1 = data.frame(tau = data2$tau, causal_neural_tau = cate_causalnetwork$V1)
results_in_sim2 = c()
results_in_sim2_uplift = c()
results_in_sim2_revenue = c() 

dp_clipping = function(CATE, min_CATE, max_CATE, epsilon){
  set.seed(1)
  clip <- function(x, a, b) {
    ifelse(x < a, a, ifelse(x > b, b, x))
  }
  clipped = clip(CATE, a =min_CATE, b = max_CATE)
  sensitivity = max_CATE - min_CATE
  clipped = DPpack::LaplaceMechanism(clipped, epsilon, sensitivity)
  return(clipped)
}

percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  print(percent)
  top = floor(100000 * percent)
  second_1$customer = 1:100000
  second_1$selection_tau = 0
  second_1$selection_true = 0
  second_1$selection_tau_3 = 0
  second_1$selection_tau_1 = 0
  second_1$selection_tau_05 = 0
  second_1$selection_tau_001 = 0
  second_1$selection_tau_005 = 0
  second_1$selection_tau_5 = 0
  second_1$random = sample(x = c(0,1), size = 1000, replace = TRUE, prob= c(1-percent,percent))
  second_1$cost = 0.5
  second_1$selection_true[as.data.frame(sort(data2$tau, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  # now with local dp
  min_CATE = 0
  max_CATE = 10
  clipped_001 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.01)
  clipped_005 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
  clipped_05 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
  clipped_1 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
  clipped_3 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
  clipped_5 = dp_clipping(second_1$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)
  
  second_1$selection_tau[as.data.frame(sort(second_1$causal_neural_tau, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  second_1$selection_tau_001[as.data.frame(sort(clipped_001, 
                                                decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_1$selection_tau_005[as.data.frame(sort(clipped_005, 
                                            decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_1$selection_tau_05[as.data.frame(sort(clipped_05, 
                                           decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_1$selection_tau_1[as.data.frame(sort(clipped_1, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_1$selection_tau_3[as.data.frame(sort(clipped_3, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_1$selection_tau_5[as.data.frame(sort(clipped_5, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  
  overlap = second_1 %>% dplyr::select(customer, selection_true, selection_tau,selection_tau_001, selection_tau_005, selection_tau_05,
                                   selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_true, selection_tau_05)[2,2]/sum(selection_true),
              overlap_001 = table(selection_true, selection_tau_001)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, selection_tau_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, selection_tau_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, selection_tau_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, selection_tau_5)[2,2]/sum(selection_true))
  
  
  uplift = second_1 %>% dplyr::select(tau,cost, selection_true, selection_tau,selection_tau_001, selection_tau_005, 
                                  selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                  random) %>% 
    pivot_longer(c(selection_true, selection_tau,selection_tau_001, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(tau*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_sim2 = rbind(results_in_sim2, overlap)
  results_in_sim2_uplift = rbind(results_in_sim2_uplift, uplift)
}
results_in_sim2
results_in_sim2_uplift

results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau"] = "CNN"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_001"] = "e = 0.01"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_005"] = "e = 0.05"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_05"] = "e = 0.5"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_1"] = "e = 1"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_3"] = "e = 3"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_5"] = "e = 5"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_true"] = "true"

sim2_second = results_in_sim2_uplift %>% filter(name != "CNN") %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() +theme_minimal() +
  theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_gather$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6))  +
  theme(legend.position="bottom") + 
  scale_color_manual(values = c("orange", "purple", "#ddb321", "blue",  "grey", "#9ea900", "red", "darkgreen")) + annotate(
    'text',
    x = 70,
    y = 250000,
    label = 'Strongly protected targeting policies \n are now outperforming a random policy.', 
    size = 4
  ) + annotate(
    'curve',
    x = 70, # Play around with the coordinates until you're satisfied
    y = 320000,
    yend = 750000,
    xend = 65,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 70, # Play around with the coordinates until you're satisfied
    y = 320000,
    yend = 800000,
    xend = 65,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 70, # Play around with the coordinates until you're satisfied
    y = 320000,
    yend = 950000,
    xend = 65,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + annotate(
    'curve',
    x = 70, # Play around with the coordinates until you're satisfied
    y = 320000,
    yend = 1000000,
    xend = 65,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) 


# combine two plots
ggarrange(sim1_second, sim2_second, ncol =2, common.legend = TRUE, legend="bottom")



# sim 2: third strategy ---------------------------------------------------
cate_causalnetwork = read.csv("causal network/CATE_sim2_epochs_100_batch_100_folds_5.csv", header = F)
second_3 = data.frame(tau = data2$tau, causal_neural_tau = cate_causalnetwork$V1)
results_in_sim2 = c()
results_in_sim2_uplift = c()
results_in_sim2_revenue = c() 
percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  top = floor(100000 * percent)
  second_3$customer = 1:100000
  second_3$selection_tau = 0
  second_3$selection_true = 0
  second_3$selection_tau_3 = 0
  second_3$selection_tau_1 = 0
  second_3$selection_tau_05 = 0
  second_3$selection_tau_005 = 0
  second_3$selection_tau_5 = 0
  second_3$random = sample(x = c(0,1), size = 100000, replace = TRUE, prob= c(1-percent,percent))
  second_3$cost = 0.5
  second_3$selection_true[as.data.frame(sort(second_3$tau, 
                                          decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  second_3$selection_tau[as.data.frame(sort(second_3$causal_neural_tau, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = second_3$selection_tau
  epsilon_range = c(0.05,0.5,1,3,5)
  for (epsilon in epsilon_range){
    print(epsilon)
    P = matrix(nrow = 2, ncol = 2)
    diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
    P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
    
    responses = c()
    for (i in 1:length(pop)){
      #print(i)
      if(pop[i] == 0){responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob= P[1,]))}
      else{responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob=P[2,]))}
    }
    if(epsilon == 0.5){
      second_3$selection_tau_05 = responses 
      index05_0 = which(second_3$selection_tau_05 == 0)
      index05 = which(second_3$selection_tau_05 == 1)
      second_3$selection_tau_05 = 0
      if(top > length(index05)){
        second_3$selection_tau_05[sample(index05, length(index05))] = 1
        second_3$selection_tau_05[sample(index05_0, top - length(index05))] = 1
      }else{
        second_3$selection_tau_05[sample(index05, top)] = 1
      }
    } else if(epsilon == 0.05){
      second_3$selection_tau_005 = responses 
      index005_0 = which(second_3$selection_tau_005 == 0)
      index005 = which(second_3$selection_tau_005 == 1)
      second_3$selection_tau_005 = 0
      if(top > length(index005)){
        second_3$selection_tau_005[sample(index005, length(index005))] = 1
        second_3$selection_tau_005[sample(index005_0, top - length(index005))] = 1
      }else{
        second_3$selection_tau_005[sample(index005, top)] = 1
      }
    } else if(epsilon == 5){
      second_3$selection_tau_5 = responses 
      index5_0 = which(second_3$selection_tau_5 == 0)
      index5 = which(second_3$selection_tau_5 == 1)
      second_3$selection_tau_5 = 0
      if(top > length(index5)){
        second_3$selection_tau_5[sample(index5, length(index5))] = 1
        second_3$selection_tau_5[sample(index5_0, top - length(index5))] = 1
      }else{
        second_3$selection_tau_5[sample(index5, top)] = 1
      }
    } else if(epsilon == 1){
      second_3$selection_tau_1 = responses 
      index1_0 = which(second_3$selection_tau_1 == 0)
      index1 = which(second_3$selection_tau_1 == 1)
      second_3$selection_tau_1 = 0
      if(top > length(index1)){
        second_3$selection_tau_1[sample(index1, length(index1))] = 1
        second_3$selection_tau_1[sample(index1_0, top - length(index1))] = 1
      }else{
        second_3$selection_tau_1[sample(index1, top)] = 1
      }
    } else {
      second_3$selection_tau_3 = responses 
      index3_0 = which(second_3$selection_tau_3 == 0)
      index3 = which(second_3$selection_tau_3 == 1)
      second_3$selection_tau_3 = 0
      if(top > length(index3)){
        second_3$selection_tau_3[sample(index3, length(index3))] = 1
        second_3$selection_tau_3[sample(index3_0, top - length(index3))] = 1
      }else{
        second_3$selection_tau_3[sample(index3, top)] = 1
      }
    }
  }
  overlap = second_3 %>% dplyr::select(customer, selection_true, selection_tau, selection_tau_005, selection_tau_05,
                                    selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_true, selection_tau_05)[2,2]/sum(selection_true),
              overlap_005 = table(selection_true, selection_tau_005)[2,2]/sum(selection_true),
              overlap_1 = table(selection_true, selection_tau_1)[2,2]/sum(selection_true),
              overlap_3 = table(selection_true, selection_tau_3)[2,2]/sum(selection_true),
              overlap_5 = table(selection_true, selection_tau_5)[2,2]/sum(selection_true))
  uplift = second_3 %>% dplyr::select(tau,cost, selection_true, selection_tau, selection_tau_005, 
                                   selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                   random) %>% 
    pivot_longer(c(selection_true, selection_tau, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(tau*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_sim2 = rbind(results_in_sim2, overlap)
  results_in_sim2_uplift = rbind(results_in_sim2_uplift, uplift)
}
results_in_sim2
results_in_sim2_uplift

results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau"] = "CNN"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_005"] = "0.05"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_05"] = "0.5"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_1"] = "1"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_3"] = "3"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_tau_5"] = "5"
results_in_sim2_uplift$name[results_in_sim2_uplift$name == "selection_true"] = "real"

#saveRDS(results_in_sim2_uplift, "results_in_sim2_uplift.RDS")
results_in_sim2_uplift = readRDS("results_in_sim2_uplift.RDS")

uplift_plot_sim2 = results_in_sim2_uplift %>% filter(name != "CNN") %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + annotate(
    'text',
    x = 20,
    y = 1e6,
    label = 'Privacy protected policies capture \n the true targeting policy even closer.', 
    size = 4
  ) + annotate(
    'curve',
    x = 20, # Play around with the coordinates until you're satisfied
    y = 94e4,
    yend = 82e4,
    xend = 30,
    linewidth = 0.5,
    curvature = 0.2,
    arrow = arrow(length = unit(0.2, 'cm'))
  ) + theme_bw() +
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(results_in_sim2_uplift$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6)) +
  theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "#ddb321", "blue",  "grey", "#9ea900", "red", "darkgreen"))

uplift_plot_sim2


ggarrange(uplift_plot_sim1, uplift_plot_sim2, ncol =2, common.legend = TRUE, legend="bottom")

# first field experiment ---------------------------------------------------------------
data = readr::read_rds("features_privacy_.RDS")
summary(data)

#old data
data$APP_INSTALLED[is.na(data$APP_INSTALLED) == T] = 0
data$APP_USES_8_WEEKS[is.na(data$APP_USES_8_WEEKS) == T] = 0
data[is.na(data) == T] = 0
data$GENDER = as.numeric(as.factor(data$GENDER))-1
data$PROVINCE = as.numeric(as.factor(data$PROVINCE)) -1
data$MIN_SALESDATE = NULL
data$MAX_SALESDATE = NULL
data$RESP_MAILING_PERIOD_DURATION = NULL
data$REG_DATE = NULL
data = mutate_all(data, as.numeric)
data_covariates = data[,c(1:122)]
summary(data)
descr(data, stats = c("min","Q1", "mean","Q3","max"), transpose = T, style = 'rmarkdown', round.digits = 2)

summary(lm("RESP_REVENUE ~ .", data = data[,c(1:11,13:42,44:45,47:55,57:62,64:86,88:89,91:96,98:120,122)]))

#write.csv(data, "features_roughclean_privacy_.csv")

## For the field experiment, we want to target the easter mailing (to do with privacy).
# 10% control group rule of thumb
# usually four coupons in the mailing (RESP_PROMOS_REDEEMDED). individual targeting rules. NA = did not redeem.
# RESP_REVENUE = negative, consumers returned products or discount
# mailing 4th january = 6th of feb
# RESP_FREQ is the amount of times.
# goal to predict RESP_PROMOS_REDEEMDED (0/1) and optionally revenue.

## if randomization is perfect (which is attempted for the experiment) then ATT = ATE = 3.5
data %>%
  group_by(RESP_TREATED) %>%
  summarise(REVENUE = mean(RESP_REVENUE),
            REVENUE_sd = sd(RESP_REVENUE)/n(),
            n = n()) %>% kable(format = "latex")


mean(data$RESP_REVENUE[data$RESP_TREATED == 1]) - mean(data$RESP_REVENUE[data$RESP_TREATED == 0])
t.test(data$RESP_REVENUE[data$RESP_TREATED == 1], data$RESP_REVENUE[data$RESP_TREATED == 0])

summary(lm("RESP_REVENUE ~ RESP_TREATED", data = data))

## average treatment effect with regression
data_covariates = data[,c(1:122)]
summary(data_covariates)
normal_model = lm("RESP_REVENUE ~ .", data_covariates)
summary(normal_model)

## direct estimation
muhat.treat <- predict(normal_model, newdata=transform(data_covariates, RESP_TREATED=1))
muhat.ctrl <- predict(normal_model, newdata=transform(data_covariates, RESP_TREATED=0))
ate.est <- mean(muhat.treat) - mean(muhat.ctrl)
ate.est

# indirect estimation
## with regression
set.seed(1)
model_treated = lm("RESP_REVENUE ~ .", data_covariates[data_covariates$RESP_TREATED == 1,])
model_control = lm("RESP_REVENUE ~ .", data_covariates[data_covariates$RESP_TREATED == 0,])

predictions_treated = predict(model_treated, data_covariates)
predictions_control = predict(model_control, data_covariates)

CATE_lm = predictions_treated - predictions_control
mean(CATE_lm)
sd(CATE_lm)/nrow(data_covariates)

## with decision tree (HTE)
model_treated_tree = rpart(RESP_REVENUE ~ ., data_covariates[data_covariates$RESP_TREATED == 1,])
model_control_tree = rpart(RESP_REVENUE ~ ., data_covariates[data_covariates$RESP_TREATED == 0,])

predictions_treated_tree = predict(model_treated_tree, data_covariates)
predictions_control_tree = predict(model_control_tree, data_covariates)

CATE_tree = predictions_treated_tree - predictions_control_tree
mean(CATE_tree)
sd(CATE_tree)/nrow(data_covariates)

## with random forest
set.seed(1)
model_treated_rf = randomForest(RESP_REVENUE ~ ., data_covariates[data_covariates$RESP_TREATED == 1,], maxnodes = 100)
model_control_rf = randomForest(RESP_REVENUE ~ ., data_covariates[data_covariates$RESP_TREATED == 0,], maxnodes = 100)

predictions_treated_rf = predict(model_treated_rf, data_covariates)
predictions_control_rf = predict(model_control_rf, data_covariates)

CATE_rf = predictions_treated_rf - predictions_control_rf
#write.csv(CATE_rf, "CATE_rf.csv", row.names = F)
CATE_rf = read.csv("CATE_rf.csv")
mean(CATE_rf$x)
sd(CATE_rf$x)/nrow(data_covariates)

## GLMnet
set.seed(1)
resp_revenue_treated = data_covariates[data_covariates$RESP_TREATED == 1,]$RESP_REVENUE
resp_revenue_control = data_covariates[data_covariates$RESP_TREATED == 0,]$RESP_REVENUE
data_covariates_glmnet_treated = data_covariates[data_covariates$RESP_TREATED == 1,1:120]
data_covariates_glmnet_control = data_covariates[data_covariates$RESP_TREATED == 0,1:120]

# treated
x <- data_covariates_glmnet_treated
y <- resp_revenue_treated
cvfit <- cv.glmnet(as.matrix(x), y)
prediction_glmnet_treated = as.data.frame(predict(cvfit, newx = as.matrix(data_covariates[,1:120]), s = "lambda.min"))

# control
x <- data_covariates_glmnet_control
y <- resp_revenue_control
cvfit <- cv.glmnet(as.matrix(x), y)
prediction_glmnet_control = as.data.frame(predict(cvfit, newx = as.matrix(data_covariates[,1:120]), s = "lambda.min"))

CATE_glmnet = prediction_glmnet_treated$lambda.min - prediction_glmnet_control$lambda.min
#write.csv(CATE_glmnet, "CATE_glmnet.csv", row.names = F)
CATE_glmnet = read.csv("CATE_glmnet.csv")
mean(CATE_glmnet$x)
sd(CATE_glmnet$x)

## assessing heterogeneity
het_effects <- data.frame(ols = CATE_lm,
                          LASSO = CATE_glmnet$x,
                          #KNN = CATE_knn$x,
                          tree = CATE_tree,
                          random_forest = CATE_rf)
colnames(het_effects) = c("ols","LASSO", "decisiontree", "randomforest")

do.call(data.frame, 
                         list(mean = apply(het_effects, 2, mean),
                              sd = apply(het_effects, 2, sd),
                              median = apply(het_effects, 2, median),
                              min = apply(het_effects, 2, min),
                              max = apply(het_effects, 2, max)))


method_cate_indirect = rbind(data.frame(CATE = het_effects$ols, method = "ols"), data.frame(CATE = het_effects$LASSO, method = "LASSO"), data.frame(CATE = het_effects$decisiontree, method = "decision tree"), data.frame(CATE = het_effects$randomforest, method = "random forest"))
means_indirect <- aggregate(CATE ~  method, method_cate_indirect, mean)

method_cate_indirect$method = factor(method_cate_indirect$method, levels = c("ols","LASSO", "decision tree", "random forest"))

method_cate_indirect %>%
  group_by(method) %>%
  summarise(CATE = mean(CATE),
            CATE_sd = sd(CATE),
            n = n())

method_cate_indirect %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() + geom_text_repel(data = means_indirect, box.padding = 0.5, size  = 4.5, aes(label = round(CATE,2))) + theme_minimal(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13)) + ylab("tau(X) (in euros)") + xlab("")

# Propensity & IPW --------------------------------------------------------------------
## propensity
## Did randomization go well?
## estimate the propensity to treatment. we need to assume functional form and estimator to explain treatment. 
## if randomization is successful most of them are insignificant.
model_treatment = glm("RESP_TREATED ~ .", data = data_covariates[1:121], family = "binomial")
summary(model_treatment)

## predict the treatment propensity
data$treatment_propensity = predict(model_treatment, data, type = "response")
summary(data$treatment_propensity)

## plot the propensity to treatment of treatment group and control group.
data$RESP_TREATED = as.factor(data$RESP_TREATED)
data %>% ggplot(aes(x = treatment_propensity, color = as.factor(RESP_TREATED))) + geom_density(alpha=0.7) + theme_minimal() + xlim(min(data$treatment_propensity),max(data$treatment_propensity)) + labs(color = "Treatment (Coupon)") + scale_color_grey(start = 0, end = 0.7) + xlab("e^(x)") + theme(legend.position="bottom") + xlab(TeX("$\\hat{e}(X_i)$")) + scale_color_manual(values = c("black", "red")) #data %>% ggplot(aes(x = treatment_propensity)) + facet_wrap(~RESP_TREATED) + geom_density(alpha=0.2) + theme_minimal() + xlim(min(data$treatment_propensity),max(data$treatment_propensity))

data %>% ggplot(aes(x = treatment_propensity, y = RESP_TREATED, color = as.factor(RESP_TREATED), lineype = as.factor(RESP_TREATED))) + geom_boxplot() + theme_bw(base_size = 13) + labs(color = "Treatment (Coupon)") + ylab("treatment status (received coupon = 1)") + xlab("estimated propensity scores") + scale_color_manual(values = c("black", "red")) + theme(legend.position="none", axis.text=element_text(size = 13, colour="black")) + scale_x_continuous(breaks = c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1), limits = c(0,1))

data$RESP_TREATED = (as.numeric(data$RESP_TREATED)-1)

## IPW
u_1 = ((data$RESP_TREATED) * data$RESP_REVENUE)/data$treatment_propensity
u_0 = ((1-(data$RESP_TREATED)) * data$RESP_REVENUE)/(1 - data$treatment_propensity)
CATE_ipw = u_1 - u_0
mean(CATE_ipw) # 2.411093

## AIPW
D = predictions_treated - predictions_control
R_1 = ((data$RESP_TREATED)/data$treatment_propensity)*(data$RESP_REVENUE - predictions_treated)
R_0 = ((1 - data$RESP_TREATED)/(1 - data$treatment_propensity)) * (data$RESP_REVENUE - predictions_control) 

AIPW = mean(D) + mean(R_1) - mean(R_0)
AIPW # 2.480321


# direct estimation methods -----------------------------------------------
# causal forest
X = data_covariates[,c(1:120)]
Y = data_covariates$RESP_REVENUE
W = data_covariates$RESP_TREATED

causalforest <- grf::causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = 100,
  seed = 1
)

# cate from random forest
CATE_causalforest <- predict(causalforest, newdata = X, type = "vector")

#write.csv(CATE_causalforest$predictions, "CATE_causal_forest.csv", row.names = F)
CATE_causalforest = read.csv("CATE_causal_forest.csv")
mean(CATE_causalforest$x)

# cate from causal network
CATE_causal_network = read.csv("networks/CATE.csv", header = F)
data$CATE_causal_network = read.csv("networks/CATE.csv", header = F)

mean(CATE_causal_network$V1)
sd(CATE_causal_network$V1)

# r-learner (rboost)
CATE_boost = rlearner::rboost(x = as.matrix(X), w = as.matrix(W), y = as.matrix(Y), k_folds = 5, nthread=1, verbose = TRUE)
CATE_boost_predictions = predict(CATE_boost, as.matrix(X))
mean(CATE_boost_predictions)
write.csv(CATE_boost_predictions, "CATE_rboost.csv", row.names = F)

# r-learner (rlasso)
CATE_lasso = rlearner::rlasso(x = as.matrix(X), w = as.matrix(W), y = as.matrix(Y), k_folds = 5)
CATE_lasso_predictions = predict(CATE_lasso, as.matrix(X))
#write.csv(CATE_lasso_predictions, "CATE_rlasso.csv", row.names = F)
CATE_lasso_predictions = read.csv("CATE_rlasso.csv")
CATE_lasso_predictions = as.numeric(CATE_lasso_predictions$V1)
mean(CATE_lasso_predictions)
sd(CATE_lasso_predictions)

## assessing heterogeneity
het_effects <- data.frame(rlasso = CATE_lasso_predictions,
                          #rboost = CATE_boost_predictions,
                          causalforest = CATE_causalforest$x,
                          causalnetwork = CATE_causal_network$V1)

colnames(het_effects) = c("rlasso", "causalforest", "causalneuralnetwork")

do.call(data.frame, 
        list(mean = apply(het_effects, 2, mean),
             sd = apply(het_effects, 2, sd),
             median = apply(het_effects, 2, median),
             min = apply(het_effects, 2, min),
             max = apply(het_effects, 2, max)))


method_cate = rbind(data.frame(CATE = het_effects$rlasso, method = "rlasso"), data.frame(CATE = het_effects$causalforest, method = "causal forest"), data.frame(CATE = het_effects$causalneuralnetwork, method = "CNN"))

method_cate$method = factor(method_cate$method, levels = c("rlasso","causal forest", "CNN"))

method_cate %>%
  group_by(method) %>%
  summarise(CATE = mean(CATE),
            n = n())

method_cate %>% ggplot(aes(x = CATE, y = method)) + geom_violin() +  geom_text_repel(data = means, box.padding = 1.5, size  = 4.5, aes(label = round(CATE,2))) + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text.x = element_text(size = 13)) + ylab("estimated tau(X) (in euros)") + xlab("") + coord_flip()

# combine plots
method_cate$estimator = "direct"
method_cate_indirect$estimator = "indirect"
cates = rbind(method_cate,method_cate_indirect)
means <- aggregate(CATE ~  method, method_cate, mean)
cates %>% ggplot(aes(x = method, y = CATE)) + geom_boxplot() +  geom_text_repel(data = means, box.padding = 1.5, size  = 4.5, aes(label = round(CATE,2))) + facet_wrap(~estimator, scales = "free")+ theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text.x = element_text(size = 13)) + ylab("estimated tau(X) (in euros)") + xlab("")

# with epsilon ------------------------------------------------------------
#CATE_0_02 = read.csv("CATE_0.02.csv", header = F) 
data$CATE_0_05 = read.csv("CATE_estimates_0_05_tuning_1.csv", header = F)$V1
#CATE_0_1 = read.csv("CATE_0.1.csv", header = F) 
data$CATE_0_5 = read.csv("CATE_estimates_0_5_tuning_1.csv", header = F)$V1
#CATE_0_75 = read.csv("networks/CATE_0.75.csv", header = F)
#CATE_1 = read.csv("networks/CATE_1.0.csv", header = F) 
#CATE_3 = read.csv("networks/CATE_3.0.csv", header = F) 
data$CATE_5 = read.csv("CATE_estimates_5_tuning.csv", header = F)$V1
#CATE_7 = read.csv("networks/CATE_7.0.csv", header = F)
#CATE_9 = read.csv("networks/CATE_9.0.csv", header = F)
#CATE_13 = read.csv("networks/CATE_12.97.csv", header = F)
data$CATE_50 = read.csv("CATE_estimates_50_tuning_1.csv", header = F)$V1
#CATE_150 = read.csv("networks/CATE_149.65.csv", header = F)
data$CATE_500 = read.csv("CATE_estimates_500_tuning.csv", header = F)$V1
data$CATE_5000 = read.csv("CATE_estimates_5000_tuning_1.csv", header = F)$V1
data$CATE_50000 = read.csv("CATE_estimates_50000_tuning.csv", header = F)$V1
data$CATE_100000 = read.csv("CATE_estimates_100000_tuning.csv", header = F)$V1
data$CATE_500000 = read.csv("CATE_estimates_500000_tuning_1.csv", header = F)$V1

data$CATE_causal_network = read.csv("networks/CATE.csv", header = F)$V1

eps_cate = rbind(data.frame(CATE = data$CATE_0_05, eps = "e = 0.05", obs = 1:length(data$CATE_0_05)), 
                 data.frame(CATE = data$CATE_0_5, eps = "e = 0.5", obs = 1:length(data$CATE_0_5)), 
                 data.frame(CATE = data$CATE_5, eps = "e = 5", obs = 1:length(data$CATE_5)), 
                 data.frame(CATE = data$CATE_50, eps = "e = 50", obs = 1:length(data$CATE_50)), 
                 data.frame(CATE = data$CATE_500, eps = "e = 500", obs = 1:length(data$CATE_500)),
                 data.frame(CATE = data$CATE_5000, eps = "e = 5000", obs = 1:length(data$CATE_5000)),
                 data.frame(CATE = data$CATE_50000, eps = "e = 50000", obs = 1:length(data$CATE_50000)),
                 data.frame(CATE = data$CATE_100000, eps = "e = 100000", obs = 1:length(data$CATE_100000)),
                 data.frame(CATE = data$CATE_500000, eps = "e = 500000", obs = 1:length(data$CATE_500000)),
                 data.frame(CATE = data$CATE_causal_network, eps = "CNN", obs = 1:length(data$CATE_causal_network)))
eps_cate$eps = factor(eps_cate$eps, levels = c("e = 0.05", "e = 0.5", "e = 5","e = 50", "e = 500", "e = 5000", "e = 50000","e = 100000", "e = 500000","CNN")) 

eps_cate %>%
  group_by(eps) %>%
  summarise(CATE_mean = mean(CATE),
            CATE_sd = sd(CATE))

means <- aggregate(CATE ~  eps, eps_cate, mean)
eps_cate %>% ggplot(aes(x= eps, y= CATE)) + geom_text_repel(data = means, box.padding = 1.5, vjust = 1, size  = 4.5, 
                                                            aes(label = round(CATE,2)))  + geom_boxplot() + theme_minimal() + 
  theme_minimal(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), 
                                        axis.text.x = element_text(size = 13)) + ylab("tau(X) (in euros)") + 
  xlab("privacy risk (epsilon)")+ theme(legend.position = "none") + scale_y_continuous(breaks = c(0,1,2,3,4,5,10,15,20))

# second field experiment results -------------------------------------------------
response = read.csv("evaluation_export/response.csv")
input_features = read.csv("evaluation_export/input_features.csv")
features = read.csv("evaluation_export/mailing_period_features.csv")
features[is.na(features)] = 0
summary(input_features)
summary(response)
summary(features)

colnames(input_features[3:122]) == colnames(data_covariates[1:120])
#write.csv(input_features[3:122], "X_out_of_sample.csv", row.names = F)
getwd()
features$DAYS_SINCE_REG_MAILING_PERIOD = features$DAYS_SINCE_REG
features$AGE_BUCKET_MAILING_PERIOD = features$AGE_BUCKET
features$GENDER_MAILING_PERIOD = features$GENDER
features$PROVINCE_MAILING_PERIOD = features$PROVINCE
features$APP_INSTALLED_MAILING_PERIOD = features$APP_INSTALLED

features$DAYS_SINCE_REG = NULL
features$AGE_BUCKET = NULL
features$GENDER = NULL
features$PROVINCE = NULL
features$APP_INSTALLED = NULL

jo = left_join(response, input_features)
jo = left_join(jo, features, by = "FAKE_ID")
colnames(jo)

summary(jo[,5:29])

jo %>% group_by(GROUP) %>%
  summarise(REVENUE_sum = sum(REVENUE),
            REVENUE_mean = mean(REVENUE),
            REVENUE_sd = sd(REVENUE)/n(),
            n = n(),
            probability = n/391289)

jo %>% group_by(TREATED) %>%
  summarise(REVENUE_sum = sum(REVENUE),
            REVENUE_mean = mean(REVENUE),
            REVENUE_sd = sd(REVENUE)/n(),
            n = n())

# constant cost of 0.5 for 
jo$profit = jo$REVENUE + jo$DISCOUNT_OECARD_MAILING_PERIOD - 0.5 # .50 cents is cost for coupon
jo$MAX_SALESDATE = NULL
jo$MIN_SALESDATE = NULL

# add epsilon out-of-sample
jo$CATE_0_05 = read.csv("CATE_estimates_0_05_tuning_out.csv", header = F)$V1
jo$CATE_0_5 = read.csv("CATE_estimates_0_5_tuning_out.csv", header = F)$V1
jo$CATE_5 = read.csv("CATE_estimates_5_tuning_out.csv", header = F)$V1
jo$CATE_50 = read.csv("CATE_estimates_50_tuning_out.csv", header = F)$V1
jo$CATE_500 = read.csv("CATE_estimates_500_tuning_out.csv", header = F)$V1
jo$CATE_5000 = read.csv("CATE_estimates_5000_tuning_out_1.csv", header = F)$V1
jo$CATE_50000 = read.csv("CATE_estimates_50000_tuning_out.csv", header = F)$V1
jo$CATE_100000 = read.csv("CATE_estimates_100000_tuning_out.csv", header = F)$V1
jo$CATE_500000 = read.csv("CATE_estimates_500000_tuning_out.csv", header = F)$V1

summary(jo$CATE_0_05)
colnames(jo)

#treatment_covariates = jo[,c(14,16:27,36:43,49:51,62:91,93:95,106:135,199)]
#summary(lm("profit ~ .", data = treatment_covariates))

## randomization
#library(nnet)
#library(MASS)
#colnames(treatment_covariates)
#multi_prop = multinom(GROUP ~ ., data=treatment_covariates)

#propensity = as.data.frame(multi_prop$fitted.values)  %>% pivot_longer(cols = c(CONTROL, TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50))

#propensity$name[propensity$name == "CONTROL"] = "control"
#propensity$name[propensity$name == "TAU_HAT"] = "true"
#propensity$name[propensity$name == "TAU_HAT_EPSILON_0_05"] = "0.05"
#propensity$name[propensity$name == "TAU_HAT_EPSILON_0_5"] = "0.5"
#propensity$name[propensity$name == "TAU_HAT_EPSILON_5"] = "5"
#propensity$name[propensity$name == "TAU_HAT_EPSILON_50"] = "50"

#propensity %>% group_by(name) %>% summarize(mean(value))

#propensity %>% ggplot(aes(x = value, y = name)) + geom_boxplot() + theme_bw() + theme_bw(base_size = 14) + theme(text = element_text(size = 14), axis.text = element_text(size = 14, color = "black"), strip.text.x = element_text(size = 14), axis.text.x = element_text(size = 14, color = 'black'),legend.text=element_text(size=14, color = 'black'))

#jo %>% group_by(GROUP) %>% summarize(n = n(),
#                                     prob = n()/391289)

# plot out of sample predictions
plot_out_of_sample = jo %>% dplyr::select(profit, TAU_HAT,CATE_500000, CATE_100000,CATE_5000,CATE_50000, CATE_500, CATE_50, CATE_5, CATE_0_5, CATE_0_05, GROUP) %>% pivot_longer(cols = c("TAU_HAT", "CATE_500000","CATE_50000", "CATE_100000","CATE_5000", "CATE_500", "CATE_50", "CATE_5", "CATE_0_5", "CATE_0_05")) 
plot_out_of_sample$name[plot_out_of_sample$name == "TAU_HAT"] = "CNN"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_500000"] = "e = 500000"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_100000"] = "e = 100000"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_50000"] = "e = 50000"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_5000"] = "e = 5000"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_500"] = "e = 500"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_50"] = "e = 50"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_5"] = "e = 5"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_0_5"] = "e = 0.5"
plot_out_of_sample$name[plot_out_of_sample$name == "CATE_0_05"] = "e = 0.05"
plot_out_of_sample = plot_out_of_sample %>% dplyr::select(value, name)
colnames(plot_out_of_sample) = c("CATE", "eps")
eps_cate = eps_cate %>% dplyr::select(CATE, eps)

eps_cate$sample = "in-sample"
plot_out_of_sample$sample = "out-of-sample"
in_out_sample = rbind(eps_cate, plot_out_of_sample)
colnames(in_out_sample)
in_out_sample %>% group_by(eps, sample) %>% summarize(mean(CATE))

in_out_sample = in_out_sample %>% filter(eps == "e = 0.05" | eps == "e = 0.5" | eps == "e = 5" | eps == "e = 50" | eps == "CNN")
means <- aggregate(CATE ~  eps + sample, in_out_sample, mean)
in_out_sample %>% 
  ggplot(aes(x= eps, y= CATE, color = sample, shape = sample, linetype = sample)) + geom_boxplot(width = 0.5) + geom_point() +
  geom_text_repel(data = means, position = position_dodge(width = .9), vjust = -0.5, size  = 5, 
                  aes(label = round(CATE,2)), show.legend = FALSE) + 
  scale_color_manual(values= c("black", "red"))+ theme_bw(base_size = 13) + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), 
        axis.text.x = element_text(size = 13), legend.text=element_text(size=13)) + 
  ylab("estimated tau(X) (in euros)") + xlab("privacy risk (epsilon)") + theme(legend.position="bottom")

# mate --------------------------------------------------------------------
# in sample
data$causal_neural_network = read.csv("networks/CATE.csv", header = F)$V1
mate = data %>% dplyr::select(PARTNER_REDEEMED_12_MONTHS, DAYS_SINCE_REG, AGE_BUCKET, GENDER, causal_neural_network, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500, CATE_5000,CATE_500000) %>% pivot_longer(c(causal_neural_network, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500, CATE_5000, CATE_500000))
unique(mate$name)
mate$name[mate$name == "causal_forest"] = "causal forest"
mate$name[mate$name == "causal_neural_network"] = "CNN"
#mate %>% ggplot(aes(x = DAYS_SINCE_REG, y = value)) + geom_point() + geom_smooth() + facet_wrap(~name, scales = "free_y") + ylab("tau(X)") + xlab("Tenure (in days)") + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13))

# out of sample
mate_out_of_sample = jo %>% dplyr::select(PARTNER_REDEEMED_12_MONTHS, DAYS_SINCE_REG, AGE_BUCKET, GENDER, TAU_HAT, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500, CATE_5000,CATE_500000) %>% pivot_longer(c(TAU_HAT,  TAU_HAT, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500, CATE_5000,CATE_500000))
mate$sample = "in-sample"
mate_out_of_sample$sample = "out-of-sample"
mate = rbind(mate, mate_out_of_sample)

mate$name[mate$name == "causal_neural_network"] = "CNN"
mate$name[mate$name == "TAU_HAT"] = "CNN"
mate$name[mate$name == "CATE_500000"] = "e = 500000"
mate$name[mate$name == "CATE_50000"] = "e = 50000"
mate$name[mate$name == "CATE_5000"] = "e = 5000"
mate$name[mate$name == "CATE_500"] = "e = 500"
mate$name[mate$name == "CATE_50"] = "e = 50"
mate$name[mate$name == "CATE_5"] = "e = 5"
mate$name[mate$name == "CATE_0_5"] = "e = 0.5"
mate$name[mate$name == "CATE_0_05"] = "e = 0.05"
unique(mate$name)

mate$name = factor(mate$name, levels = c("e = 0.05", "e = 0.5", "e = 5", "e = 50", "e = 500","e = 5000","e = 50000","e = 500000", "CNN"))

testing = mate %>% pivot_longer(c(PARTNER_REDEEMED_12_MONTHS , DAYS_SINCE_REG), names_repair = "minimal")
colnames(testing) = c("AGE_BUCKET", "GENDER", "method", "tau", "sample", "name", "value")


testing$name[testing$name == "DAYS_SINCE_REG"] = "Tenure (in days)"
testing$name[testing$name == "PARTNER_REDEEMED_12_MONTHS"] = "Coupon Redemption"

#summary(lm("tau ~ sample + method + name",testing))

#plots
tenure= testing %>% filter(method != "e = 500") %>% filter(method != "e = 5000") %>% filter(method != "e = 500000") %>% filter(name == "Tenure (in days)") %>% sample_n(size = 100000) %>% ggplot(aes(x = value, y = tau, color = sample, shape = sample)) + geom_point(size = 2) + facet_grid(sample~method, scales = "free") + scale_color_manual(values = c("black", "red")) + ylab("estimated tau(X) (in euros)") + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13)) + xlab("Tenure (in days)") +theme(legend.position="bottom") +theme(legend.text=element_text(size=13)) + scale_x_continuous(breaks = c(0, 1000, 1500))
redemption = testing %>% filter(method != "e = 500") %>% filter(method != "e = 5000") %>% filter(method != "e = 500000") %>% filter(name == "Coupon Redemption") %>% sample_n(size = 200000) %>% ggplot(aes(x = value, y = tau, color = sample, shape = sample)) + geom_point(size = 2) + facet_grid(sample~method, scales = "free") + scale_color_manual(values = c("black", "red")) + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13)) + xlab("Coupon Redemption (in #)") +theme(legend.position="bottom") +theme(legend.text=element_text(size=13)) + ylab("") + scale_x_continuous(breaks = c(0, 25, 50))

g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}

mylegend<-g_legend(tenure)

grid.arrange(arrangeGrob(tenure + theme(legend.position="none"),
                         redemption + theme(legend.position="none"),
                         nrow=1), mylegend, nrow = 2,heights=c(10, 1))

# first privacy protection strategy ------------------------------------------------------
selection = function(percentage = percentage, data = jo){
  top = floor(nrow(data) * percentage)
  jo$selection_tau = 0
  jo$selection_tau_005 = 0
  jo$selection_tau_05 = 0
  jo$selection_tau_5 = 0
  jo$selection_tau_50 = 0
  jo$selection_tau_500 = 0
  jo$selection_tau_5000 = 0
  jo$selection_tau_50000 = 0
  jo$selection_tau_100000 = 0
  jo$selection_tau_500000 = 0
  jo$selection_revenue = 0
  jo$random = 0
  jo$selection_tau[as.data.frame(sort(jo$TAU_HAT, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_005[as.data.frame(sort(jo$CATE_0_05, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_05[as.data.frame(sort(jo$CATE_0_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_5[as.data.frame(sort(jo$CATE_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_50[as.data.frame(sort(jo$CATE_50, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_500[as.data.frame(sort(jo$CATE_500, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_5000[as.data.frame(sort(jo$CATE_5000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_50000[as.data.frame(sort(jo$CATE_50000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_100000[as.data.frame(sort(jo$CATE_100000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_500000[as.data.frame(sort(jo$CATE_500000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_revenue[as.data.frame(sort(jo$REVENUE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$random[sample(1:nrow(jo), top)] = 1
  jo$cost = 0.5
  revenue = jo %>% select(REVENUE, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_100000,selection_tau_500000, selection_revenue, random)) %>% group_by(name) %>% summarize(revenue = sum(REVENUE*value))
  uplift = jo %>% select(TAU_HAT,cost, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random)) %>% group_by(name) %>% summarize(revenue = (sum(TAU_HAT*value) - sum(cost*value)))
  
  results <- list("uplift" = uplift, "revenue" = revenue)
return(results)}

revenue_gather = c()
uplift_gather = c()
results = c()
for (percent in c(0.0001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99, 0.999)){
  print(percent)
  results = selection(percentage = percent, data = jo)
  uplift = results$uplift
  uplift$percentage = percent
  
  results$revenue
  revenue = results$revenue
  revenue$percentage = percent
  #revenue_gather = rbind(revenue_gather, revenue)
  uplift_gather = rbind(uplift_gather, uplift)
}

uplift_gather %>% ggplot(aes(x = percentage, y = revenue, color = name, shape = name)) + geom_point(size = 2.5) +geom_line() + theme_minimal() + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(angle = 0, size = 13), strip.text = element_text(size = 13)) +
  scale_shape_manual(values = 1:length(unique(uplift_gather$name)))


# another way to calculate the same thing
jo$TAU_HAT_estimate = jo$TAU_HAT
jo$cost = 0.5
jo %>% select(TAU_HAT_estimate,cost, TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE) %>% pivot_longer(c(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE)) %>%  
  group_by(name) %>% 
  top_n(value, n = floor(nrow(jo) * 0.95)) %>% 
  slice_head(n = floor(nrow(jo) * 0.95)) %>% 
  summarize(sum(TAU_HAT_estimate) - sum(cost))


# to add: assume a profit margin 
uplift_gather$name[uplift_gather$name == "selection_revenue"] = "revenue"
uplift_gather$name[uplift_gather$name == "selection_tau"] = "CNN"
uplift_gather$name[uplift_gather$name == "selection_tau_005"] = "e = 0.05"
uplift_gather$name[uplift_gather$name == "selection_tau_05"] = "e = 0.5"
uplift_gather$name[uplift_gather$name == "selection_tau_5"] = "e = 5"
uplift_gather$name[uplift_gather$name == "selection_tau_50"] = "e = 50"
uplift_gather$name[uplift_gather$name == "selection_tau_500"] = "e = 500"
uplift_gather$name[uplift_gather$name == "selection_tau_5000"] = "e = 5000"
uplift_gather$name[uplift_gather$name == "selection_tau_50000"] = "e = 50000"
uplift_gather$name[uplift_gather$name == "selection_tau_100000"] = "e = 100000"
uplift_gather$name[uplift_gather$name == "selection_tau_500000"] = "e = 500000"
uplift_gather$name = factor(uplift_gather$name, levels = c("random","e = 500000","e = 100000","e = 5000","e = 50000","e = 500", "e = 0.5", "revenue", "e = 5", "CNN", "e = 50", "e = 0.05"))
uplift_gather %>% ggplot(aes(x = percentage*100, y = revenue, color = name, shape = name)) + geom_point(size = 3) + geom_line(size = 1) + theme_minimal() + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy", c("random", "e = 0.5", "revenue", "e = 5", "CNN", "e = 50", "e = 0.05"),values = 1:length(unique(uplift_gather$name))) + theme(legend.text=element_text(size=13)) + 
  ylab("profit (in euros)") + xlab("top % targeted") + guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10000,25000,50000,100000,150000,200000,225000, max(uplift_gather$revenue))) + theme(legend.position="bottom") 

# in sample
colnames(data)
data$causal_neural_network = read.csv("networks/CATE.csv", header = F)$V1
selection = function(percentage, data = data){
  top = floor(nrow(data) * percentage)
  data$selection_tau = 0
  data$selection_tau_005 = 0
  data$selection_tau_05 = 0
  data$selection_tau_5 = 0
  data$selection_tau_50 = 0
  data$selection_tau_500 = 0
  data$selection_tau_5000 = 0
  data$selection_tau_50000 = 0
  data$selection_tau_100000 = 0
  data$selection_tau_500000 = 0
  data$selection_revenue = 0
  data$random = 0
  data$selection_tau[data$causal_neural_network %in% sort(data$causal_neural_network, decreasing = TRUE)[1:top]][1:top] = 1
  data$selection_tau_005[as.data.frame(sort(data$CATE_0_05, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_05[as.data.frame(sort(data$CATE_0_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_5[as.data.frame(sort(data$CATE_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_50[as.data.frame(sort(data$CATE_50, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_500[as.data.frame(sort(data$CATE_500, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_5000[as.data.frame(sort(data$CATE_5000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_50000[as.data.frame(sort(data$CATE_50000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_100000[as.data.frame(sort(data$CATE_100000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_500000[as.data.frame(sort(data$CATE_500000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_revenue[as.data.frame(sort(data$RESP_REVENUE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$random[sample(1:nrow(data), top)] = 1
  data$cost = 0.5
  revenue = data %>% select(RESP_REVENUE, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000, selection_tau_500000, selection_revenue, random) %>% pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random)) %>% group_by(name) %>% summarize(revenue = sum(RESP_REVENUE*value))
  uplift = data %>% select(causal_neural_network,cost, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000, selection_tau_500000, selection_revenue, random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random)) %>% group_by(name) %>% summarize(revenue = (sum(causal_neural_network*value) - sum(cost*value)))
  
  results <- list("uplift" = uplift, "revenue" = revenue)
  return(results)}

revenue_gather_in_sample = c()
uplift_gather_in_sample = c()
results = c()
for (percent in c(0.0001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99, 0.999)){
  print(percent)
  results = selection(percentage = percent, data = data)
  uplift = results$uplift
  uplift$percentage = percent
  
  results$revenue
  revenue = results$revenue
  revenue$percentage = percent
  revenue_gather_in_sample = rbind(revenue_gather_in_sample, revenue)
  uplift_gather_in_sample = rbind(uplift_gather_in_sample, uplift)
}
uplift_gather_in_sample %>% group_by(name) %>% summarize(mean = mean(revenue)) 
uplift_gather %>% group_by(name) %>% summarize(mean = mean(revenue))

# to add: assume a profit margin 
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_revenue"] = "revenue"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau"] = "CNN"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_005"] = "e = 0.05"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_05"] = "e = 0.5"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_5"] = "e = 5"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_50"] = "e = 50"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_500"] = "e = 500"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_5000"] = "e = 5000"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_50000"] = "e = 50000"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_100000"] = "e = 100000"
uplift_gather_in_sample$name[uplift_gather_in_sample$name == "selection_tau_500000"] = "e = 500000"
uplift_gather_in_sample$name = factor(uplift_gather_in_sample$name, levels = c("random", "e = 0.5", "revenue", "e = 5", "CNN", "e = 50","e = 500","e = 5000","e = 50000","e = 100000","e = 500000", "e = 0.05"))

uplift_gather_in_sample %>%
  ggplot(aes(x = percentage*100, y = revenue, color = name, shape = name)) + geom_point(size = 3) + geom_line(size = 1) + theme_minimal() + theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy", c("random", "e = 0.5", "revenue", "e = 5", "CNN", "e = 50", "e = 0.05"),values = 1:length(unique(uplift_gather_in_sample$name))) + theme(legend.text=element_text(size=13)) + 
  ylab("profit (in euros)") + xlab("top % targeted") + guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +
  scale_y_continuous(labels = scales::comma, breaks = c(10000,50000,100000,200000,300000,400000,500000, 600000,700000, max(uplift_gather_in_sample$revenue), max(uplift_gather$revenue))) + theme(legend.position="bottom") 

uplift_gather$sample = "out-of-sample"
uplift_gather_in_sample$sample = "in-sample"
uplift_sort = rbind(uplift_gather, uplift_gather_in_sample)
uplift_sort$name = as.character(uplift_sort$name)

library(scales)
uplift_sort = uplift_sort %>% filter(name != "e = 100000") %>% filter(name != "e = 5000") %>% 
  filter(name != "e = 500") %>% filter(name != "e = 50000")
uplift_sort$name = factor(uplift_sort$name, levels = c("e = 0.05","e = 0.5","e = 5", "e = 50", "e = 500000", "random", "revenue", "CNN"))
uplift_sort %>% 
  ggplot(aes(x = percentage*100, y = revenue, color = name)) + geom_point(size = 3) + geom_line(size = 1) +
  theme_bw(base_size = 13) + facet_wrap(~sample, scales = "free_y") + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), 
                                                                            axis.text.x = element_text(size = 13), 
                                                                            strip.text = element_text(size = 13)) + 
  #scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_sort$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("estimated profit (in thousands of euros)") +scale_x_log10()+ xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3), 
                     breaks = c(10000,100000,200000,300000, 400000, 500000, 600000, 700000, 800000, max(uplift_gather$revenue))) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99))+ theme(legend.position="bottom") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black", "darkgreen")) 

#zooming in on epsilon levels
uplift_sort %>% filter(percentage < 0.10) %>% filter(name != "CNN")%>% filter(name != "e = 500000") %>% filter(name != "revenue") %>% filter(name != "random") %>% filter(name != "e = 100000") %>% filter(name != "e = 5000") %>% ggplot(aes(x = percentage*100, y = revenue, color = name, shape = name)) + geom_point(size = 3) + geom_line(size = 1) + theme_bw(base_size = 13) + facet_wrap(~sample, scales = "free_y") + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(uplift_sort$name))) + theme(legend.text=element_text(size=13)) + 
  ylab("estimated profit (in thousands of euros)") +scale_x_log10()+ xlab("top % targeted") + guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3), breaks = c(1000,10000,20000,50000,75000,100000,150000,200000,300000, 400000, 500000, 600000, 700000, 800000, max(uplift_gather$revenue))) + scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99))+ theme(legend.position="bottom") +
  scale_color_manual(values = c("black", "blue",  "grey", "#ddb321", "#954010", "#9ea900", "darkgreen", "#576396", "red", "darkred")) + scale_y_log10()

# dml versus revenue sorting
uplift_sort %>% group_by(name, sample) %>% filter(sample == "in-sample") %>% summarize(meanprofit = mean(revenue),
                                                     meanprofitloss = mean(revenue) - 496317) %>% filter(grepl("e = ",name))

uplift_sort %>% group_by(name, sample) %>% filter(sample == "out-of-sample") %>% summarize(meanprofit = mean(revenue),
                                                                                       meanprofit = (mean(revenue) - 155111)) %>% filter(grepl("e = ",name))

# realized revenue --------------------------------------------------------
# insample
data$cost = 0.5
data$causal_neural_network = read.csv("networks/CATE.csv", header = F)$V1
data$random = sample(1:nrow(data))

cor(data$causal_neural_network, data$RESP_REVENUE)

cor(jo$TAU_HAT, jo$REVENUE)

revenue_gather_in_sample = c()
for (percentage in c(0.0001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99, 0.999)){
  print(percentage)
  revenue = data %>% select(RESP_REVENUE,cost, causal_neural_network, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500,CATE_5000,CATE_50000,CATE_100000,CATE_500000, random) %>% 
    pivot_longer(c(causal_neural_network, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500,CATE_5000,CATE_50000,CATE_100000,CATE_500000, random)) %>%  group_by(name) %>% 
    top_n(value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(revenue = sum(RESP_REVENUE))

  revenue$percentage = percentage
  revenue_gather_in_sample = rbind(revenue_gather_in_sample, revenue)
}
revenue_gather_in_sample

# out of sample
percentage = 0.01
top = floor(nrow(jo) * percentage)
jo$cost = 0.5
jo$random = sample(1:nrow(jo))

revenue_gather_out_sample = c()
for (percentage in c(0.0001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99, 0.999)){
  print(percentage)
  revenue = jo %>% select(REVENUE,cost, TAU_HAT, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500,CATE_5000,CATE_50000,CATE_100000,CATE_500000, random) %>% 
    pivot_longer(c(TAU_HAT, CATE_0_05, CATE_0_5, CATE_5, CATE_50, CATE_500,CATE_5000,CATE_50000,CATE_100000,CATE_500000, random)) %>%  group_by(name) %>% 
    top_n(value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(revenue = sum(REVENUE))
  revenue$percentage = percentage
  revenue_gather_out_sample = rbind(revenue_gather_out_sample, revenue)
}
revenue_gather_out_sample


revenue_gather_out_sample$sample = "out-of-sample"
revenue_gather_in_sample$sample = "in-sample"
revenue_sort = rbind(revenue_gather_out_sample, revenue_gather_in_sample)

revenue_sort$name[revenue_sort$name == "TAU_HAT"] = "CNN"
revenue_sort$name[revenue_sort$name == "CATE_0_05"] = "e = 0.05"
revenue_sort$name[revenue_sort$name == "CATE_0_5"] = "e = 0.5"
revenue_sort$name[revenue_sort$name == "CATE_5"] = "e = 5"
revenue_sort$name[revenue_sort$name == "CATE_50"] = "e = 50"
revenue_sort$name[revenue_sort$name == "CATE_500"] = "e = 500"
revenue_sort$name[revenue_sort$name == "CATE_5000"] = "e = 5000"
revenue_sort$name[revenue_sort$name == "CATE_50000"] = "e = 50000"
revenue_sort$name[revenue_sort$name == "CATE_100000"] = "e = 100000"
revenue_sort$name[revenue_sort$name == "CATE_500000"] = "e = 500000"
revenue_sort$name[revenue_sort$name == "causal_neural_network"] = "CNN"

library(scales)
revenue_sort = revenue_sort %>%filter(name != "e = 100000") %>% filter(name != "e = 5000") %>% filter(name != "e = 50000") %>% filter(name != "e = 500") 
revenue_sort$name = factor(revenue_sort$name, levels = c("e = 0.05","e = 0.5","e = 5", "e = 50", "e = 500000", "random", "revenue", "CNN"))
revenue_sort %>% ggplot(aes(x = percentage*100, y = revenue, color = name)) + geom_point(size = 3) + 
  geom_line(size = 1) + theme_bw(base_size = 13) + facet_wrap(~sample, scales = "free_y") + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), 
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  #scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(revenue_sort$name))) + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profits (in millions of euros)") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6), 
                     breaks = c(1e6, 5e6, 7.5e6,10e6,15e6,20e6,11042131,
                                max(revenue_sort$revenue))) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "darkgreen"))

revenue_sort %>% group_by(sample) %>% summarize(max(revenue))
revenue_sort %>% group_by(name, sample) %>% summarize(mean = mean(revenue)) 

# overlap -----------------------------------------------------------------
selection = function(percentage, data){
  top = nrow(data) * percent
  jo$customer = 1:nrow(jo)
  jo$selection_tau = 0
  jo$selection_tau_005 = 0
  jo$selection_tau_05 = 0
  jo$selection_tau_5 = 0
  jo$selection_tau_50 = 0
  jo$selection_tau_500 = 0
  jo$selection_tau_5000 = 0
  jo$selection_tau_50000 = 0
  jo$selection_tau_100000 = 0
  jo$selection_tau_500000 = 0
  jo$selection_revenue = 0
  jo$random = 0
  jo$selection_tau[jo$TAU_HAT %in% sort(jo$TAU_HAT, decreasing = TRUE)[1:top]][1:top] = 1
  jo$selection_tau_005[as.data.frame(sort(jo$CATE_0_05, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_05[as.data.frame(sort(jo$CATE_0_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_5[as.data.frame(sort(jo$CATE_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_50[as.data.frame(sort(jo$CATE_50, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_500[as.data.frame(sort(jo$CATE_500, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_5000[as.data.frame(sort(jo$CATE_5000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_50000[as.data.frame(sort(jo$CATE_50000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_100000[as.data.frame(sort(jo$CATE_100000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_tau_500000[as.data.frame(sort(jo$CATE_500000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$selection_revenue[as.data.frame(sort(jo$REVENUE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  jo$random[sample(1:nrow(jo), top)] = 1
  overlap = jo %>% select(customer, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau),
              overlap_50 = table(selection_tau, selection_tau_50)[2,2]/sum(selection_tau),
              overlap_500 = table(selection_tau, selection_tau_500)[2,2]/sum(selection_tau),
              overlap_5000 = table(selection_tau, selection_tau_5000)[2,2]/sum(selection_tau),
              overlap_50000 = table(selection_tau, selection_tau_50000)[2,2]/sum(selection_tau),
              overlap_100000 = table(selection_tau, selection_tau_100000)[2,2]/sum(selection_tau),
              overlap_500000 = table(selection_tau, selection_tau_500000)[2,2]/sum(selection_tau),
              overlap_revenue = table(selection_tau, selection_revenue)[2,2]/sum(selection_tau),
              total = sum(selection_tau))
  overlap$percent = percentage
  results <- list("overlap" = overlap)
return(results)}


# 0.0001,0.01,0.015,0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,0.075,0.1,0.15,0.175,0.2, 0.225,0.25, 0.275,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99, 0.999,1
overlap_gather_out_sample = c()
for (percent in c(0.0001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)){
  print(percent)
  results = selection(percentage = percent, data = data)
  overlap = results$overlap
  overlap_gather_out_sample = rbind(overlap_gather_out_sample, overlap)
}


# inofsample
colnames(data)
data$causal_neural_network = read.csv("networks/CATE.csv", header = F)$V1
selection = function(percentage = 0.1, data = data){
  top = floor(nrow(data) * percentage)
  data$customer = 1:nrow(data)
  data$selection_tau = 0
  data$selection_tau_005 = 0
  data$selection_tau_05 = 0
  data$selection_tau_5 = 0
  data$selection_tau_50 = 0
  data$selection_tau_500 = 0
  data$selection_tau_5000 = 0
  data$selection_tau_50000 = 0
  data$selection_tau_100000 = 0
  data$selection_tau_500000 = 0
  data$selection_revenue = 0
  data$random = 0
  data$selection_tau[data$causal_neural_network %in% sort(data$causal_neural_network, decreasing = TRUE)[1:top]][1:top] = 1
  data$selection_tau_005[as.data.frame(sort(data$CATE_0_05, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_05[as.data.frame(sort(data$CATE_0_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_5[as.data.frame(sort(data$CATE_5, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_50[as.data.frame(sort(data$CATE_50, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_500[as.data.frame(sort(data$CATE_500, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_5000[as.data.frame(sort(data$CATE_5000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_50000[as.data.frame(sort(data$CATE_50000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_100000[as.data.frame(sort(data$CATE_100000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_tau_500000[as.data.frame(sort(data$CATE_500000, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$selection_revenue[as.data.frame(sort(data$RESP_REVENUE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  data$random[sample(1:nrow(data), top)] = 1
  overlap = data %>% select(customer, selection_tau, selection_tau_005, selection_tau_05, selection_tau_5, selection_tau_50,selection_tau_500,selection_tau_5000,selection_tau_50000,selection_tau_100000,selection_tau_500000, selection_revenue, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau),
              overlap_50 = table(selection_tau, selection_tau_50)[2,2]/sum(selection_tau),
              overlap_500 = table(selection_tau, selection_tau_500)[2,2]/sum(selection_tau),
              overlap_5000 = table(selection_tau, selection_tau_5000)[2,2]/sum(selection_tau),
              overlap_50000 = table(selection_tau, selection_tau_50000)[2,2]/sum(selection_tau),
              overlap_100000 = table(selection_tau, selection_tau_100000)[2,2]/sum(selection_tau),
              overlap_500000 = table(selection_tau, selection_tau_500000)[2,2]/sum(selection_tau),
              overlap_revenue = table(selection_tau, selection_revenue)[2,2]/sum(selection_tau),
              total = sum(selection_tau))
  overlap$percent = percentage
  results <- list("overlap" = overlap)
  return(results)}

overlap_gather_in_sample = c()
for (percent in c(0.0001,0.01, 0.03,0.1,0.15,0.175,0.2, 0.225,0.25, 0.275,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)){
  print(percent)
  results = selection(percentage = percent, data = data)
  overlap = results$overlap
  overlap_gather_in_sample = rbind(overlap_gather_in_sample, overlap)
}

overlap_gather_out_sample$sample = "out-of-sample"
overlap_gather_in_sample$sample = "in-sample"
overlap = rbind(overlap_gather_out_sample, overlap_gather_in_sample)
overlap = overlap %>% pivot_longer(c(overlap_random, overlap_005, overlap_05, overlap_5, overlap_50,overlap_500,
                                     overlap_5000,overlap_50000,overlap_100000,overlap_500000, overlap_revenue)) 
overlap$name[overlap$name == "overlap_revenue"] = "revenue"
overlap$name[overlap$name == "overlap_random"] = "random"
overlap$name[overlap$name == "overlap_005"] = "e = 0.05"
overlap$name[overlap$name == "overlap_05"] = "e = 0.5"
overlap$name[overlap$name == "overlap_5"] = "e = 5"
overlap$name[overlap$name == "overlap_50"] = "e = 50"
overlap$name[overlap$name == "overlap_500"] = "e = 500"
overlap$name[overlap$name == "overlap_5000"] = "e = 5000"
overlap$name[overlap$name == "overlap_50000"] = "e = 50000"
overlap$name[overlap$name == "overlap_100000"] = "e = 100000"
overlap$name[overlap$name == "overlap_500000"] = "e = 500000"

overlap = overlap %>%filter(name != "e = 100000") %>% filter(name != "e = 5000") %>% filter(name != "e = 50000") %>%
  filter(name != "e = 500") 
overlap %>% 
  ggplot(aes(x = percent*100, y = value*100, color = as.factor(name))) + 
  geom_point(size = 3) + geom_line(size = 1) + ylab("overlap (in %)") + xlab("top % targeted") + 
  facet_wrap(~sample) + guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +  
  theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
                                   axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  #scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(overlap$name))) + 
  theme(legend.text=element_text(size=12)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90, 99)) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom", legend.box="vertical") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black"))

soverlap %>% filter(sample == "in-sample") %>% 
  group_by(name) %>% summarize(mean_overlap = mean(value)) %>% arrange(mean_overlap)


# second privacy protection strategy ---------------------------------------
## second privacy strategy (in-sample)
RESP_REVENUE = data$RESP_REVENUE
CATE_causal_network = read.csv("networks/CATE.csv", header = F)
third = data.frame(causal_neural_tau = CATE_causal_network$V1, revenue = RESP_REVENUE)
results_in_field = c()
results_in_field_uplift = c()
results_in_field_revenue = c() 

dp_clipping = function(CATE, min_CATE, max_CATE, epsilon){
  set.seed(1)
  clip <- function(x, a, b) {
    ifelse(x < a, a, ifelse(x > b, b, x))
  }
  clipped = clip(CATE, a =min_CATE, b = max_CATE)
  sensitivity = max_CATE - min_CATE
  clipped = DPpack::LaplaceMechanism(clipped, epsilon, sensitivity)
  return(clipped)
}

percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  print(percent)
  top = floor(nrow(third) * percent)
  third$customer = 1:nrow(third)
  third$selection_revenue = 0
  third$selection_tau = 0
  third$selection_true = 0
  third$selection_tau_3 = 0
  third$selection_tau_1 = 0
  third$selection_tau_05 = 0
  third$selection_tau_005 = 0
  third$selection_tau_5 = 0
  third$random = sample(x = c(0,1), size = nrow(CATE_causal_network), replace = TRUE, prob= c(1-percent,percent))
  third$cost = 0.5
  
  # now with local dp
  min_CATE = 0
  max_CATE = 5
  clipped_005 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
  clipped_05 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
  clipped_1 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
  clipped_3 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
  clipped_5 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)
  
  third$selection_tau[as.data.frame(sort(third$causal_neural_tau, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  third$selection_revenue[as.data.frame(sort(RESP_REVENUE, 
                                             decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  third$selection_tau_005[as.data.frame(sort(clipped_005, 
                                             decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third$selection_tau_05[as.data.frame(sort(clipped_05, 
                                            decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third$selection_tau_1[as.data.frame(sort(clipped_1, 
                                           decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third$selection_tau_3[as.data.frame(sort(clipped_3, 
                                           decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third$selection_tau_5[as.data.frame(sort(clipped_5, 
                                           decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  
  overlap = third %>% dplyr::select(customer, selection_tau,selection_revenue, selection_tau_005, selection_tau_05,
                                    selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_1 = table(selection_tau, selection_tau_1)[2,2]/sum(selection_tau),
              overlap_3 = table(selection_tau, selection_tau_3)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau),
              overlap_revenue = table(selection_tau, selection_revenue)[2,2]/sum(selection_tau))
  
  
  uplift = third %>% dplyr::select(causal_neural_tau ,cost, selection_tau, selection_revenue, selection_tau_005, 
                                   selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                   random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_revenue, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(causal_neural_tau *value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_field = rbind(results_in_field, overlap)
  results_in_field_uplift = rbind(results_in_field_uplift, uplift)
}
results_in_field
results_in_field_uplift

results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau"] = "CNN"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau_005"] = "0.05"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau_05"] = "0.5"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau_1"] = "1"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau_3"] = "3"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_tau_5"] = "5"
results_in_field_uplift$name[results_in_field_uplift$name == "selection_revenue"] = "revenue"

#saveRDS(results_in_field_uplift,"results_in_field_uplift.RDS")
#saveRDS(results_in_field,"results_in_field.RDS")
results_in_field_uplift = readRDS("results_in_field_uplift.RDS")
results_in_field = readRDS("results_in_field.RDS")

results_in_field_uplift$name = factor(results_in_field_uplift$name, levels = c("e = 0.05", "e = 0.5", "e = 1", "e = 3", "e = 5", "random", "revenue", "CNN"))
profit_in_field = results_in_field_uplift %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() +theme_minimal() +
  theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13))  + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6))  +
  theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black", "darkgreen"))   +
  annotate('text',x = 70,y = 150000,label = 'The profit levels have improved \n compared to the previous strategy.', size = 4) + 
  annotate('curve',x = 60, y = 200000,yend = 640000,xend = 40,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm'))) +
  annotate('curve',x = 60, y = 200000,yend = 575000,xend = 40,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm'))) +
  annotate('curve',x = 60, y = 200000,yend = 450000,xend = 40,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))+
  annotate('curve',x = 60, y = 200000,yend = 390000,xend = 40,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))

profit_in_field

# overlap
results_in_field = results_in_field %>% pivot_longer(cols = c(overlap_random,overlap_005, overlap_05, overlap_1, overlap_3,overlap_5, overlap_revenue))
results_in_field$name[results_in_field$name == "overlap_005"] = "e = 0.05"
results_in_field$name[results_in_field$name == "overlap_05"] = "e = 0.5"
results_in_field$name[results_in_field$name == "overlap_1"] = "e = 1"
results_in_field$name[results_in_field$name == "overlap_3"] = "e = 3"
results_in_field$name[results_in_field$name == "overlap_5"] = "e = 5"
results_in_field$name[results_in_field$name == "overlap_revenue"] = "revenue"
results_in_field$name[results_in_field$name == "overlap_random"] = "random"

overlap_in_field = results_in_field %>% 
  ggplot(aes(x = percentage*100, y = value*100, color = name)) + geom_point(size = 2.5) +geom_line() + ylab("overlap (in %)") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +  
  theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(overlap$name))) + theme(legend.text=element_text(size=12)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90, 99)) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom", legend.box="vertical") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "darkgreen", "red", "black")) + 
  annotate('text',x = 70,y = 20,label = 'Privacy protected policies outperform \n managerial heuristic (revenue policy) \n and random policy.', size = 4) + 
  annotate('curve',x = 70, y = 30,yend = 63,xend = 50,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm'))) +
  annotate('curve',x = 70, y = 30,yend = 76,xend = 50,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm'))) +
  annotate('curve',x = 70, y = 30,yend = 83,xend = 50,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))+
  annotate('curve',x = 70, y = 30,yend = 49,xend = 50,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))

overlap_in_field

# CATE distribution
min_CATE = 0
max_CATE = 5
clipped_005 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
clipped_05 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
clipped_1 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
clipped_3 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
clipped_5 = dp_clipping(third$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)
clipped_005_out = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
clipped_05_out = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
clipped_1_out = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
clipped_3_out = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
clipped_5_out = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)

strat2_CATE = bind_cols(clipped_005 = clipped_005, clipped_05 = clipped_05, clipped_1 = clipped_1, clipped_3 = clipped_3, clipped_5 = clipped_5, CNN = third$causal_neural_tau, sample = "in-sample")
strat2_out_CATE = bind_cols(clipped_005_out = clipped_005_out, clipped_05_out = clipped_05_out, clipped_1_out = clipped_1_out, clipped_3_out = clipped_3_out, clipped_5_out = clipped_5_out, CNN = third_out$causal_neural_tau, sample = "out-of-sample")
strat2_CATE = strat2_CATE %>% pivot_longer(cols = c(clipped_005, clipped_05, clipped_1, clipped_3, clipped_5, CNN))
strat2_out_CATE = strat2_out_CATE %>% pivot_longer(cols = c(clipped_005_out, clipped_05_out, clipped_1_out, clipped_3_out, clipped_5_out, CNN))
strat2_CATE = rbind(strat2_CATE, strat2_out_CATE)

strat2_CATE$name[strat2_CATE$name == "clipped_05"] = "e = 0.5"
strat2_CATE$name[strat2_CATE$name == "clipped_005"] = "e = 0.05"
strat2_CATE$name[strat2_CATE$name == "clipped_1"] = "e = 1"
strat2_CATE$name[strat2_CATE$name == "clipped_3"] = "e = 3"
strat2_CATE$name[strat2_CATE$name == "clipped_5"] = "e = 5"
strat2_CATE$name[strat2_CATE$name == "clipped_05_out"] = "e = 0.5"
strat2_CATE$name[strat2_CATE$name == "clipped_005_out"] = "e = 0.05"
strat2_CATE$name[strat2_CATE$name == "clipped_1_out"] = "e = 1"
strat2_CATE$name[strat2_CATE$name == "clipped_3_out"] = "e = 3"
strat2_CATE$name[strat2_CATE$name == "clipped_5_out"] = "e = 5"

means <- aggregate(value ~ name + sample, strat2_CATE, mean)

strat2_CATE$name = factor(strat2_CATE$name, levels = c("e = 0.05", "e = 0.5", "e = 1", "e = 3","e = 5", "CNN"))

strat2_CATE %>%
  ggplot(aes(x = name, y = value, color = sample)) + geom_boxplot(width = 0.5) +
  geom_text_repel(data = means, position = position_dodge(width = .9), size  = 5, box.padding = 0.5, 
                  aes(label = round(value,2)), show.legend = FALSE) + 
  scale_color_manual(values= c("black", "red"))+ theme_bw(base_size = 13) + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), 
        axis.text.x = element_text(size = 13), legend.text=element_text(size=13)) + 
  ylab("estimated tau(X) (in euros)") + xlab("privacy risk (epsilon)") + theme(legend.position="bottom") + ylim(-100,100)

# out of sample
RESP_REVENUE = jo$REVENUE
third_out = data.frame(causal_neural_tau = jo$TAU_HAT, revenue = RESP_REVENUE)
results_out_field = c()
results_out_field_uplift = c()
results_out_field_revenue = c() 

percentage = c(0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  print(percent)
  top = floor(nrow(third_out) * percent)
  third_out$customer = 1:nrow(third_out)
  third_out$selection_revenue = 0
  third_out$selection_tau = 0
  third_out$selection_true = 0
  third_out$selection_tau_3 = 0
  third_out$selection_tau_1 = 0
  third_out$selection_tau_05 = 0
  third_out$selection_tau_005 = 0
  third_out$selection_tau_5 = 0
  third_out$random = sample(x = c(0,1), size = nrow(third_out), replace = TRUE, prob= c(1-percent,percent))
  third_out$cost = 0.5
  
  # now with local dp
  min_CATE = 0
  max_CATE = 5
  clipped_005 = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.05)
  clipped_05 = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 0.5)
  clipped_1 = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 1)
  clipped_3 = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 3)
  clipped_5 = dp_clipping(third_out$causal_neural_tau, min_CATE = min_CATE, max_CATE = max_CATE, epsilon = 5)
  
  third_out$selection_tau[as.data.frame(sort(third_out$causal_neural_tau, 
                                             decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  third_out$selection_revenue[as.data.frame(sort(RESP_REVENUE, 
                                                 decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  third_out$selection_tau_005[as.data.frame(sort(clipped_005, 
                                                 decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third_out$selection_tau_05[as.data.frame(sort(clipped_05, 
                                                decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third_out$selection_tau_1[as.data.frame(sort(clipped_1, 
                                               decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third_out$selection_tau_3[as.data.frame(sort(clipped_3, 
                                               decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  third_out$selection_tau_5[as.data.frame(sort(clipped_5, 
                                               decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  
  overlap = third_out %>% dplyr::select(customer, selection_tau,selection_revenue, selection_tau_005, selection_tau_05,
                                        selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_1 = table(selection_tau, selection_tau_1)[2,2]/sum(selection_tau),
              overlap_3 = table(selection_tau, selection_tau_3)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau),
              overlap_revenue = table(selection_tau, selection_revenue)[2,2]/sum(selection_tau))
  
  
  uplift = third_out %>% dplyr::select(causal_neural_tau ,cost, selection_tau, selection_revenue, selection_tau_005, 
                                       selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                       random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_revenue, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(causal_neural_tau *value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_out_field = rbind(results_out_field, overlap)
  results_out_field_uplift = rbind(results_out_field_uplift, uplift)
}
results_out_field
results_out_field_uplift

#saveRDS(results_out_field_uplift,"results_out_field_uplift.RDS")
results_out_field_uplift = readRDS("results_out_field_uplift.RDS")

results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau"] = "CNN"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau_005"] = "e = 0.05"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau_05"] = "e = 0.5"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau_1"] = "e = 1"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau_3"] = "e = 3"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_tau_5"] = "e = 5"
results_out_field_uplift$name[results_out_field_uplift$name == "selection_revenue"] = "revenue"

library(scales)

results_out_field_uplift$name = factor(results_out_field_uplift$name, levels = c("e = 0.05", "e = 0.5", "e = 1", "e = 3", "e = 5", "random", "revenue", "CNN"))
profit_out_field = results_out_field_uplift  %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() +theme_minimal() +
  theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + 
  theme(legend.text=element_text(size=13)) + xlab("top % targeted") + ylab("") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,5e4,10e4,15e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,max(results_out_field_uplift$profit)))  +
  theme(legend.position="bottom") + 
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black", "darkgreen"))  

profit_out_field

results_out_field = results_out_field %>% pivot_longer(cols = c(overlap_random,overlap_005, overlap_05, overlap_1, overlap_3,overlap_5, overlap_revenue))
results_out_field$name[results_out_field$name == "overlap_005"] = "e = 0.05"
results_out_field$name[results_out_field$name == "overlap_05"] = "e = 0.5"
results_out_field$name[results_out_field$name == "overlap_1"] = "e = 1"
results_out_field$name[results_out_field$name == "overlap_3"] = "e = 3"
results_out_field$name[results_out_field$name == "overlap_5"] = "e = 5"
results_out_field$name[results_out_field$name == "overlap_revenue"] = "revenue"
results_out_field$name[results_out_field$name == "overlap_random"] = "random"

overlap_out_field = results_out_field %>% 
  ggplot(aes(x = percentage*100, y = value*100, color = name)) + geom_point(size = 2.5) +geom_line() + ylab("overlap (in %)") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +  
  theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(overlap$name))) + theme(legend.text=element_text(size=12)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90, 99)) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom", legend.box="vertical") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey",  "darkgreen", "red", "black")) + 
  annotate('text',x = 40,y = 90,label = 'Overlap with targeting policy \n without privacy protection increases \n for higher privacy risk.', size = 4) + 
  annotate('curve',x = 25, y = 80,yend = 60,xend = 25,linewidth = 0.5,curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))

ggarrange(overlap_in_field, overlap_out_field, ncol =2, common.legend = TRUE, legend="bottom")
ggarrange(profit_in_field, profit_out_field, ncol =2, common.legend = TRUE, legend="bottom")



# third privacy protection strategy ---------------------------------------
# insample
summary(data)
results_in_in_sample = c()
results_in_in_sample_uplift = c()
results_in_in_sample_revenue = c() 
percentage = c(0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  print(percent)
  top = floor(nrow(data) * percent)
  data$customer = 1:nrow(data)
  data$selection_tau = 0
  data$selection_tau_3 = 0
  data$selection_tau_1 = 0
  data$selection_tau_05 = 0
  data$selection_tau_005 = 0
  data$selection_tau_5 = 0
  data$random = sample(x = c(0,1), size = nrow(data), replace = TRUE, prob= c(1-percent,percent))
  data$cost = 0.5
  data$selection_tau[as.data.frame(sort(data$causal_neural_network, 
                                         decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = data$selection_tau
  epsilon_range = c(0.05,0.5,1,3,5)
  for (epsilon in epsilon_range){
    #print(epsilon)
    P = matrix(nrow = 2, ncol = 2)
    diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
    P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
    
    responses = c()
    for (i in 1:length(pop)){
      #print(i)
      if(pop[i] == 0){responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob= P[1,]))}
      else{responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob=P[2,]))}
    }
    if(epsilon == 0.5){
      data$selection_tau_05 = responses 
      index05_0 = which(data$selection_tau_05 == 0)
      index05 = which(data$selection_tau_05 == 1)
      data$selection_tau_05 = 0
      if(top > length(index05)){
        data$selection_tau_05[sample(index05, length(index05))] = 1
        data$selection_tau_05[sample(index05_0, top - length(index05))] = 1
      }else{
        data$selection_tau_05[sample(index05, top)] = 1
      }
    } else if(epsilon == 0.05){
      data$selection_tau_005 = responses 
      index005_0 = which(data$selection_tau_005 == 0)
      index005 = which(data$selection_tau_005 == 1)
      data$selection_tau_005 = 0
      if(top > length(index005)){
        data$selection_tau_005[sample(index005, length(index005))] = 1
        data$selection_tau_005[sample(index005_0, top - length(index005))] = 1
      }else{
        data$selection_tau_005[sample(index005, top)] = 1
      }
    } else if(epsilon == 5){
      data$selection_tau_5 = responses 
      index5_0 = which(data$selection_tau_5 == 0)
      index5 = which(data$selection_tau_5 == 1)
      data$selection_tau_5 = 0
      if(top > length(index5)){
        data$selection_tau_5[sample(index5, length(index5))] = 1
        data$selection_tau_5[sample(index5_0, top - length(index5))] = 1
      }else{
        data$selection_tau_5[sample(index5, top)] = 1
      }
    } else if(epsilon == 1){
      data$selection_tau_1 = responses 
      index1_0 = which(data$selection_tau_1 == 0)
      index1 = which(data$selection_tau_1 == 1)
      data$selection_tau_1 = 0
      if(top > length(index1)){
        data$selection_tau_1[sample(index1, length(index1))] = 1
        data$selection_tau_1[sample(index1_0, top - length(index1))] = 1
      }else{
        data$selection_tau_1[sample(index1, top)] = 1
      }
    } else {
      data$selection_tau_3 = responses 
      index3_0 = which(data$selection_tau_3 == 0)
      index3 = which(data$selection_tau_3 == 1)
      data$selection_tau_3 = 0
      if(top > length(index3)){
        data$selection_tau_3[sample(index3, length(index3))] = 1
        data$selection_tau_3[sample(index3_0, top - length(index3))] = 1
      }else{
        data$selection_tau_3[sample(index3, top)] = 1
      }
    }
  }
  overlap = data %>% dplyr::select(customer, selection_tau, selection_tau_005, selection_tau_05,
                                    selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_1 = table(selection_tau, selection_tau_1)[2,2]/sum(selection_tau),
              overlap_3 = table(selection_tau, selection_tau_3)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau))
  
  uplift = data %>% dplyr::select(causal_neural_network, cost, selection_tau, selection_tau_005, 
                                   selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                   random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(causal_neural_network*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_in_sample = rbind(results_in_in_sample, overlap)
  results_in_in_sample_uplift = rbind(results_in_in_sample_uplift, uplift)
}
results_in_in_sample
results_in_in_sample_uplift

#saveRDS(results_in_in_sample,"results_in_in_sample.RDS")
#saveRDS(results_in_in_sample_uplift,"results_in_in_sample_uplift.RDS")
results_in_in_sample_uplift = readRDS("results_in_in_sample_uplift.RDS")

results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau"] = "CNN"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau_005"] = "0.05"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau_05"] = "0.5"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau_1"] = "1"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau_3"] = "3"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_tau_5"] = "5"
results_in_in_sample_uplift$name[results_in_in_sample_uplift$name == "selection_true"] = "real"

results_in_in_sample_uplift$name = factor(results_in_in_sample_uplift$name, levels = c("0.05", "0.5", "1", "3", "5", "random", "revenue", "CNN"))

results_in_in_sample_uplift = results_in_in_sample_uplift %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13))  + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6))  +
  theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black", "darkgreen"))   +
  annotate('text',x = 70,y = 150000,label = 'The profit levels have improved \n compared to the previous strategy.', size = 4)

# overlap 

results_in_in_sample = readRDS("results_in_in_sample.RDS")
results_in_in_sample = results_in_in_sample %>% pivot_longer(cols = c(overlap_random,overlap_005, overlap_05, overlap_1, overlap_3,overlap_5))
results_in_in_sample$name[results_in_in_sample$name == "overlap_005"] = "e = 0.05"
results_in_in_sample$name[results_in_in_sample$name == "overlap_05"] = "e = 0.5"
results_in_in_sample$name[results_in_in_sample$name == "overlap_1"] = "e = 1"
results_in_in_sample$name[results_in_in_sample$name == "overlap_3"] = "e = 3"
results_in_in_sample$name[results_in_in_sample$name == "overlap_5"] = "e = 5"
results_in_in_sample$name[results_in_in_sample$name == "overlap_revenue"] = "revenue"
results_in_in_sample$name[results_in_in_sample$name == "overlap_random"] = "random"

overlap_in_field = results_in_in_sample %>% 
  ggplot(aes(x = percentage*100, y = value*100, color = name)) + geom_point(size = 2.5) +geom_line() + ylab("overlap (in %)") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +  
  theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(overlap$name))) + theme(legend.text=element_text(size=12)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90, 99)) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom", legend.box="vertical") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey",  "darkgreen", "red", "black")) + 
  annotate('text',x = 40,y = 80, 
           label = 'The overlap has once again increased, \n which drives the increase in profit.', 
           size = 4) + 
  annotate('curve', x = 40, y = 85, yend = 92, xend = 35, linewidth = 0.5, 
           curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm'))) + 
  annotate('curve', x = 40, y = 85, yend = 97, xend = 35, linewidth = 0.5, 
           curvature = 0.2,arrow = arrow(length = unit(0.2, 'cm')))
overlap_in_field


# out of sample
results_in_out_sample = c()
results_in_out_sample_uplift = c()
results_in_out_sample_revenue = c() 
percentage = c(0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99)
for (percent in percentage){
  print(percent)
  top = floor(nrow(jo) * percent)
  jo$customer = 1:nrow(jo)
  jo$selection_tau = 0
  jo$selection_tau_3 = 0
  jo$selection_tau_1 = 0
  jo$selection_tau_05 = 0
  jo$selection_tau_005 = 0
  jo$selection_tau_5 = 0
  jo$random = sample(x = c(0,1), size = nrow(jo), replace = TRUE, prob= c(1-percent,percent))
  jo$cost = 0.5
  jo$selection_tau[as.data.frame(sort(jo$TAU_HAT, 
                                        decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = jo$selection_tau
  epsilon_range = c(0.05,0.5,1,3,5)
  for (epsilon in epsilon_range){
    #print(epsilon)
    P = matrix(nrow = 2, ncol = 2)
    diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
    P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
    
    responses = c()
    for (i in 1:length(pop)){
      #print(i)
      if(pop[i] == 0){responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob= P[1,]))}
      else{responses = rbind(responses, sample(x = c(1:2)-1,size = 1,prob=P[2,]))}
    }
    if(epsilon == 0.5){
      jo$selection_tau_05 = responses 
      index05_0 = which(jo$selection_tau_05 == 0)
      index05 = which(jo$selection_tau_05 == 1)
      jo$selection_tau_05 = 0
      if(top > length(index05)){
        jo$selection_tau_05[sample(index05, length(index05))] = 1
        jo$selection_tau_05[sample(index05_0, top - length(index05))] = 1
      }else{
        jo$selection_tau_05[sample(index05, top)] = 1
      }
    } else if(epsilon == 0.05){
      jo$selection_tau_005 = responses 
      index005_0 = which(jo$selection_tau_005 == 0)
      index005 = which(jo$selection_tau_005 == 1)
      jo$selection_tau_005 = 0
      if(top > length(index005)){
        jo$selection_tau_005[sample(index005, length(index005))] = 1
        jo$selection_tau_005[sample(index005_0, top - length(index005))] = 1
      }else{
        jo$selection_tau_005[sample(index005, top)] = 1
      }
    } else if(epsilon == 5){
      jo$selection_tau_5 = responses 
      index5_0 = which(jo$selection_tau_5 == 0)
      index5 = which(jo$selection_tau_5 == 1)
      jo$selection_tau_5 = 0
      if(top > length(index5)){
        jo$selection_tau_5[sample(index5, length(index5))] = 1
        jo$selection_tau_5[sample(index5_0, top - length(index5))] = 1
      }else{
        jo$selection_tau_5[sample(index5, top)] = 1
      }
    } else if(epsilon == 1){
      jo$selection_tau_1 = responses 
      index1_0 = which(jo$selection_tau_1 == 0)
      index1 = which(jo$selection_tau_1 == 1)
      jo$selection_tau_1 = 0
      if(top > length(index1)){
        jo$selection_tau_1[sample(index1, length(index1))] = 1
        jo$selection_tau_1[sample(index1_0, top - length(index1))] = 1
      }else{
        jo$selection_tau_1[sample(index1, top)] = 1
      }
    } else {
      jo$selection_tau_3 = responses 
      index3_0 = which(jo$selection_tau_3 == 0)
      index3 = which(jo$selection_tau_3 == 1)
      jo$selection_tau_3 = 0
      if(top > length(index3)){
        jo$selection_tau_3[sample(index3, length(index3))] = 1
        jo$selection_tau_3[sample(index3_0, top - length(index3))] = 1
      }else{
        jo$selection_tau_3[sample(index3, top)] = 1
      }
    }
  }
  overlap = jo %>% dplyr::select(customer, selection_tau, selection_tau_005, selection_tau_05,
                                   selection_tau_1,selection_tau_3,selection_tau_5, random) %>% 
    summarize(overlap_random = table(selection_tau, random)[2,2]/sum(selection_tau),
              overlap_05 = table(selection_tau, selection_tau_05)[2,2]/sum(selection_tau),
              overlap_005 = table(selection_tau, selection_tau_005)[2,2]/sum(selection_tau),
              overlap_1 = table(selection_tau, selection_tau_1)[2,2]/sum(selection_tau),
              overlap_3 = table(selection_tau, selection_tau_3)[2,2]/sum(selection_tau),
              overlap_5 = table(selection_tau, selection_tau_5)[2,2]/sum(selection_tau))
  
  uplift = jo %>% dplyr::select(TAU_HAT, cost, selection_tau, selection_tau_005, 
                                  selection_tau_05,selection_tau_1,selection_tau_3,selection_tau_5, 
                                  random) %>% 
    pivot_longer(c(selection_tau, selection_tau_005, selection_tau_05,selection_tau_1,
                   selection_tau_3,selection_tau_5, random)) %>% 
    group_by(name) %>% summarize(profit = (sum(TAU_HAT*value) - sum(cost*value)))
  
  
  overlap$percentage = percent
  uplift$percentage = percent
  
  results_in_out_sample = rbind(results_in_out_sample, overlap)
  results_in_out_sample_uplift = rbind(results_in_out_sample_uplift, uplift)
}
results_in_out_sample
results_in_out_sample_uplift

#saveRDS(results_in_out_sample,"results_in_out_sample.RDS")
#saveRDS(results_in_out_sample_uplift,"results_in_out_sample_uplift.RDS")
results_in_out_sample_uplift = readRDS("results_in_out_sample_uplift.RDS")

results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau"] = "CNN"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau_005"] = "0.05"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau_05"] = "0.5"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau_1"] = "1"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau_3"] = "3"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_tau_5"] = "5"
results_in_out_sample_uplift$name[results_in_out_sample_uplift$name == "selection_true"] = "real"

results_in_out_sample_uplift$name = factor(results_in_out_sample_uplift$name, levels = c("0.05", "0.5", "1", "3", "5", "random", "revenue", "CNN"))

results_in_out_sample_uplift = results_in_out_sample_uplift %>%
  ggplot(aes(x = percentage*100, y = profit, color = name)) + geom_point(size = 2.5) +geom_line() + theme_bw() + 
  theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"),
        axis.text.x = element_text(size = 13), strip.text = element_text(size = 13))  + 
  theme(legend.text=element_text(size=13)) + 
  ylab("profit") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policies"), name = guide_legend(title="Targeting Policy")) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) +
  scale_y_continuous(labels = scales::comma, breaks = c(0,10e4,20e4,30e4,40e4,50e4,6e5,7e5,8e5,9e5,1e6,1.1e6))  +
  theme(legend.position="bottom") +  
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey", "#9ea900", "red", "black"))

# overlap 
results_in_out_sample = readRDS("results_in_out_sample.RDS")
results_in_out_sample = results_in_out_sample %>% pivot_longer(cols = c(overlap_random,overlap_005, overlap_05, overlap_1, overlap_3,overlap_5))
results_in_out_sample$name[results_in_out_sample$name == "overlap_005"] = "e = 0.05"
results_in_out_sample$name[results_in_out_sample$name == "overlap_05"] = "e = 0.5"
results_in_out_sample$name[results_in_out_sample$name == "overlap_1"] = "e = 1"
results_in_out_sample$name[results_in_out_sample$name == "overlap_3"] = "e = 3"
results_in_out_sample$name[results_in_out_sample$name == "overlap_5"] = "e = 5"
results_in_out_sample$name[results_in_out_sample$name == "overlap_revenue"] = "revenue"
results_in_out_sample$name[results_in_out_sample$name == "overlap_random"] = "random"

overlap_out_field = results_in_out_sample %>% 
  ggplot(aes(x = percentage*100, y = value*100, color = name)) + geom_point(size = 2.5) +geom_line() + ylab("overlap (in %)") + xlab("top % targeted") + 
  guides(color=guide_legend(title="Targeting Policy"), name = guide_legend(title="Targeting Policy")) +  
  theme_bw(base_size = 13) + theme(text = element_text(size = 13), axis.text = element_text(size = 13, color = "black"), axis.text.x = element_text(size = 13), strip.text = element_text(size = 13)) + scale_shape_manual(name = "Targeting Policy",values = 1:length(unique(overlap$name))) + theme(legend.text=element_text(size=12)) + 
  scale_y_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90, 99)) + 
  scale_x_continuous(labels = scales::comma, breaks = c(0,10,20,30,40,50,60,70,80,90,99)) + 
  theme(legend.position="bottom", legend.box="vertical") +
  scale_color_manual(values = c("purple", "#ddb321", "blue", "grey",  "darkgreen", "red", "black"))
overlap_out_field


ggarrange(results_in_in_sample_uplift, results_in_out_sample_uplift, ncol =2, common.legend = TRUE, legend="bottom")
ggarrange(overlap_in_field, overlap_out_field, ncol =2, common.legend = TRUE, legend="bottom")

# privacy elasticity ------------------------------------------------------
response = read.csv("evaluation_export/response.csv")
input_features = read.csv("evaluation_export/input_features.csv")
features = read.csv("evaluation_export/mailing_period_features.csv")
features[is.na(features)] = 0
summary(input_features)
summary(response)
summary(features)

features$DAYS_SINCE_REG_MAILING_PERIOD = features$DAYS_SINCE_REG
features$AGE_BUCKET_MAILING_PERIOD = features$AGE_BUCKET
features$GENDER_MAILING_PERIOD = features$GENDER
features$PROVINCE_MAILING_PERIOD = features$PROVINCE
features$APP_INSTALLED_MAILING_PERIOD = features$APP_INSTALLED

features$DAYS_SINCE_REG = NULL
features$AGE_BUCKET = NULL
features$GENDER = NULL
features$PROVINCE = NULL
features$APP_INSTALLED = NULL

jo = left_join(response, input_features)
jo = left_join(jo, features, by = "FAKE_ID")
colnames(jo)


# constant cost of 0.5 for 
jo$profit = jo$REVENUE + jo$DISCOUNT_OECARD_MAILING_PERIOD - 0.5 # .50 cents is cost for coupon
jo$MAX_SALESDATE = NULL
jo$MIN_SALESDATE = NULL

# add epsilon out-of-sample
jo$CATE_0_05 = read.csv("CATE_estimates_0_05_tuning_out_1.csv", header = F)$V1
jo$CATE_0_5 = read.csv("CATE_estimates_0_5_tuning_out_1.csv", header = F)$V1
jo$CATE_5 = read.csv("CATE_estimates_5_tuning_out_1.csv", header = F)$V1
jo$CATE_50 = read.csv("CATE_estimates_50_tuning_out_1.csv", header = F)$V1
jo$CATE_500 = read.csv("CATE_estimates_500_tuning_out.csv", header = F)$V1
jo$CATE_5000 = read.csv("CATE_estimates_5000_tuning_out.csv", header = F)$V1
jo$CATE_50000 = read.csv("CATE_estimates_50000_tuning_out.csv", header = F)$V1
jo$CATE_100000 = read.csv("CATE_estimates_100000_tuning_out.csv", header = F)$V1
jo$CATE_500000 = read.csv("CATE_estimates_500000_tuning_out.csv", header = F)$V1

formula = paste(names(jo[,c(16:135)]), collapse='+')
formula_behavioral = paste(names(jo[,c(20:135)]), collapse='+')

jo$PREDICTED_UPLIFT_new = 0

jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT"] = jo$TAU_HAT[jo$GROUP == "TAU_HAT"]
mean(jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT"])
jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT_EPSILON_0_05"] = jo$CATE_0_05[jo$GROUP == "TAU_HAT_EPSILON_0_05"]
hist(jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT_EPSILON_0_05"])
jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT_EPSILON_0_5"] = jo$CATE_0_5[jo$GROUP == "TAU_HAT_EPSILON_0_5"]
jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT_EPSILON_5"] = jo$CATE_5[jo$GROUP == "TAU_HAT_EPSILON_5"]
jo$PREDICTED_UPLIFT_new[jo$GROUP == "TAU_HAT_EPSILON_50"] = jo$CATE_50[jo$GROUP == "TAU_HAT_EPSILON_50"]

# with inf
epsilon_only = jo %>% filter(GROUP != "CONTROL") %>% filter(GROUP != "TAU_HAT")
epsilon_only$epsilon[epsilon_only$GROUP == "TAU_HAT_EPSILON_50"] = 50
epsilon_only$epsilon[epsilon_only$GROUP == "TAU_HAT_EPSILON_5"] = 5
epsilon_only$epsilon[epsilon_only$GROUP == "TAU_HAT_EPSILON_0_5"] = 0.5
epsilon_only$epsilon[epsilon_only$GROUP == "TAU_HAT_EPSILON_0_05"] = 0.05


epsilon_plus_inf = jo %>% filter(GROUP != "CONTROL")
epsilon_plus_inf$epsilon[epsilon_plus_inf$GROUP == "TAU_HAT_EPSILON_50"] = 50
epsilon_plus_inf$epsilon[epsilon_plus_inf$GROUP == "TAU_HAT_EPSILON_5"] = 5
epsilon_plus_inf$epsilon[epsilon_plus_inf$GROUP == "TAU_HAT_EPSILON_0_5"] = 0.5
epsilon_plus_inf$epsilon[epsilon_plus_inf$GROUP == "TAU_HAT_EPSILON_0_05"] = 0.05
epsilon_plus_inf$epsilon[epsilon_plus_inf$GROUP == "TAU_HAT"] = 100

epsilon_plus_inf$inf[epsilon_plus_inf$GROUP != "TAU_HAT"] = 0
epsilon_plus_inf$inf[epsilon_plus_inf$GROUP == "TAU_HAT"] = 1
epsilon_plus_inf$inf = as.factor(epsilon_plus_inf$inf)

cate_model_without_cov = lm("log(PREDICTED_UPLIFT_new) ~ epsilon", data = epsilon_only)
cate_model_without_demo = lm(paste0("log(PREDICTED_UPLIFT_new) ~ epsilon + ", formula_behavioral), data = epsilon_only)
cate_model = lm(paste0("log(PREDICTED_UPLIFT_new) ~ epsilon + ", formula), data = epsilon_only)
cate_model_inf = lm(paste0("log(PREDICTED_UPLIFT_new) ~ epsilon + inf+ ", formula), data =  epsilon_plus_inf)

epsilon_plus_inf %>% ggplot(aes(x = inf, y = log(PREDICTED_UPLIFT_new))) + geom_point()
test  =epsilon_plus_inf %>% select(inf, PREDICTED_UPLIFT_new)

epsilon_plus_inf %>% group_by(inf) %>% summarize(mean(PREDICTED_UPLIFT_new))

(exp(-2.779)-1)*100

summary(cate_model_without_cov)
summary(cate_model_without_demo)
summary(cate_model)
summary(cate_model_inf)

cate_model_without_cov = lm("log(PREDICTED_UPLIFT_new) ~ epsilon", data = data)

library(stargazer)
stargazer(cate_model_without_cov,cate_model_without_demo, cate_model, cate_model_inf, title="Regression Results",
          align=TRUE, dep.var.labels=c("tau(X)","tau(X)","tau(X)", "tau(X)"),
          omit.stat=c("LL","ser","f"), no.space=TRUE)

epsilons_only %>% ggplot(aes(x = as.numeric(epsilon), y = log(PREDICTED_UPLIFT))) + geom_point() + geom_smooth(method = 'lm') + theme_minimal()

jo %>% group_by(GROUP) %>% summarize(n()/391289)

# revenue
model_control = lm(paste0("REVENUE ~", formula), data = jo %>% filter(GROUP == "control"))
summary(model_control)
jo$REVENUE_IF_NOT_TREATED = predict(model_control, jo)

# epsilon targets top x percent, and the rest is control (control is not random here.)
percentage = 0.5
treated = jo %>% select(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE) %>% pivot_longer(c(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50)) %>%  group_by(name) %>% 
  top_n(value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(REVENUE_TREATED = mean(REVENUE))
  
no_treated = jo %>% select(REVENUE_IF_NOT_TREATED, TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE) %>% pivot_longer(c(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50)) %>%  group_by(name) %>% 
  top_n(-value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(REVENUE_NOT_TREATED = mean(REVENUE_IF_NOT_TREATED))

results = left_join(treated, no_treated)

# control and treatment is random, then ordering.
percentage = 0.1

set.seed(1)
idx <- sample(seq_len(nrow(jo)), size = floor(nrow(jo)*0.5))

control <- jo[idx, ]
treatment <- jo[-idx, ]
jo %>% filter(GROUP == "control") %>% select(REVENUE) %>% summarize(mean(REVENUE))
treated = treatment %>% select(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE)  %>% pivot_longer(c(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50)) %>%  group_by(name) %>%
  top_n(value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(REVENUE_TREATED = mean(REVENUE))

no_treated = control %>% select(REVENUE_IF_NOT_TREATED, TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50, REVENUE) %>% pivot_longer(c(TAU_HAT, TAU_HAT_EPSILON_0_05, TAU_HAT_EPSILON_0_5, TAU_HAT_EPSILON_5, TAU_HAT_EPSILON_50)) %>%  group_by(name) %>% 
  top_n(-value, n = floor(nrow(jo) * percentage)) %>% slice_head(n = floor(nrow(jo) * percentage)) %>% summarize(REVENUE_NOT_TREATED = mean(REVENUE_IF_NOT_TREATED))

results = left_join(treated, no_treated)
results %>% mutate(tau = REVENUE_TREATED - REVENUE_NOT_TREATED)

## epsilon
summary(lm("REVENUE ~ epsilon", data  = jo %>% filter(GROUP != "control")))


# test --------------------------------------------------------------------
# now with local dp
pop = data$selection_tau
epsilon_range = c(0.05,0.5,1,3,5)#,0.5,1,3,5

for (epsilon in epsilon_range){
  print(epsilon)
  P = matrix(nrow = 2, ncol = 2)
  diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
  P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
  
  responses = c()
  pop_0 = data %>% select(selection_tau, customer) %>% filter(selection_tau == 0)
  pop_1 = data %>% select(selection_tau, customer) %>% filter(selection_tau == 1)
  
  pop_0$selection_tau = sample(x = c(1:2)-1,size = nrow(pop_0), prob= P[1,], replace = T)
  pop_1$selection_tau = sample(x = c(1:2)-1,size = nrow(pop_1),prob=P[2,], replace = T)
  responses = bind_rows(pop_0, pop_1) %>% arrange(customer)
  
  if(epsilon == 0.5){
    data$selection_tau_05 = responses 
    index05_0 = which(data$selection_tau_05 == 0)
    index05 = which(data$selection_tau_05 == 1)
    data$selection_tau_05 = 0
    if(top > length(index05)){
      data$selection_tau_05[sample(index05, length(index05))] = 1
      data$selection_tau_05[sample(index05_0, top - length(index05))] = 1
    }else{
      data$selection_tau_05[sample(index05, top)] = 1
    }
  } else if(epsilon == 0.05){
    data$selection_tau_005 = responses 
    index005_0 = which(data$selection_tau_005 == 0)
    index005 = which(data$selection_tau_005 == 1)
    data$selection_tau_005 = 0
    if(top > length(index005)){
      data$selection_tau_005[sample(index005, length(index005))] = 1
      data$selection_tau_005[sample(index005_0, top - length(index005))] = 1
    }else{
      data$selection_tau_005[sample(index005, top)] = 1
    }
  } else if(epsilon == 5){
    data$selection_tau_5 = responses 
    index5_0 = which(data$selection_tau_5 == 0)
    index5 = which(data$selection_tau_5 == 1)
    data$selection_tau_5 = 0
    if(top > length(index5)){
      data$selection_tau_5[sample(index5, length(index5))] = 1
      data$selection_tau_5[sample(index5_0, top - length(index5))] = 1
    }else{
      data$selection_tau_5[sample(index5, top)] = 1
    }
  } else if(epsilon == 1){
    data$selection_tau_1 = responses 
    index1_0 = which(data$selection_tau_1 == 0)
    index1 = which(data$selection_tau_1 == 1)
    data$selection_tau_1 = 0
    if(top > length(index1)){
      data$selection_tau_1[sample(index1, length(index1))] = 1
      data$selection_tau_1[sample(index1_0, top - length(index1))] = 1
    }else{
      data$selection_tau_1[sample(index1, top)] = 1
    }
  } else {
    data$selection_tau_3 = responses 
    index3_0 = which(data$selection_tau_3 == 0)
    index3 = which(data$selection_tau_3 == 1)
    data$selection_tau_3 = 0
    if(top > length(index3)){
      data$selection_tau_3[sample(index3, length(index3))] = 1
      data$selection_tau_3[sample(index3_0, top - length(index3))] = 1
    }else{
      data$selection_tau_3[sample(index3, top)] = 1
    }
  }
}
