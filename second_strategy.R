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
