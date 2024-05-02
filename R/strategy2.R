protection = function(epsilon, selection, top){
  # privacy settings
  P = matrix(nrow = 2, ncol = 2)
  diag(P) = (exp(epsilon))/(2-1+exp(epsilon))
  P[is.na(P)==T] = (1)/(2-1+exp(epsilon))
  
  # get responses
  responses = rep(NA,length(selection))
  
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
