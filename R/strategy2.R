protect = function(percent, CATE, CATE_estimates, n, epsilons = c(0.05,0.5,1,3,5)){
  top = floor(n * percent)
  selection_true = rep(NA, n)
  selection_tau = rep(NA, n)
  selection_true[as.data.frame(sort(CATE, decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  selection_tau[as.data.frame(sort(CATE_estimates,decreasing = TRUE, index.return = T))$ix[1:top]] = 1
  
  # now with local dp
  pop = selection_tau
  collection = data.frame(customer = 1:n)
  for (epsilon in epsilon_range){
    print(epsilon)
    protected_selection = protection(epsilon = epsilon, selection = CATE_estimates, top = top)
    collection = cbind(collection, protected_selection)
  }
  colnames(collection) = c("customer", paste0("", gsub("\\.", "", as.character(paste0("epsilon_", epsilon_range)))))
  collection$random = sample(x = c(0,1), size = n, replace = TRUE, prob= c(1-percent,percent))
  collection$percentage = percent
  return(collection)
}

protection = function(epsilon, selection, top){
  # privacy settings
  P = matrix(nrow = 2, ncol = 2)
  diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
  P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
  
  # get responses
  responses = rep(0,length(selection))
  
  # for every row in the responses generate protected selection based on matrix above.
  for (i in 1:length(selection)){
    responses[i] = ifelse(selection[i] == 0, sample(x = c(1:2)-1,size = 1,prob= P[1,]), sample(x = c(1:2)-1,size = 1,prob=P[2,]))
  }
  
  protected_selection = responses # make responses equal to the selection
  index_0 = which(protected_selection == 0) # select the rownumbers that are equal to 0
  index = which(protected_selection == 1) # select the rownumbers that are equal to 0
  protected_selection = rep(0,length(selection)) # set the selection to zero again
  
  if(top > length(index)){
    protected_selection[sample(index, length(index))] = 1 # sample everyone from index005
    protected_selection[sample(index_0, top - length(index))] = 1 # sample from not selected to get equal amount to top (top-length is remainder)
  }else{
    protected_selection[sample(index, top)] = 1
  }
  return(protected_selection)
}
