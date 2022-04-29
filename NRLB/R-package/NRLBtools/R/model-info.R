#' Display model information
#' 
#' @description A function used to display various levels of information for NRLB model objects and pre-built NRLB models.
#' 
#' @param models an NRLB model object.
#' @param index numeric indicating the index of the model in the NRLB model object.
#' @param nrlb.model numeric index of the pre-built NRLB model.
#' 
#' @return a data frame.
#' 
#' @details \code{model.info} provides many different levels of information depending on the number of arguments passed to it. When run with
#' no arguments, model.info returns a data frame containing the information of all the NRLB models stored in the package and used in the Rastogi, et al.
#' (2018) paper. 
#' 
#' When the index of an NRLB model (\code{nrlb.model}) is provided, a data frame containing the name, footprint size (\code{k}), mode relative affinity
#' (\code{RelativeAffinity}), and the optimal sequence (\code{BestSequence}) is provided for all modes in the model. The nonspecific binding mode is 
#' denoted by \code{NSB} and only has an associated relative affinity. 
#' 
#' When only an NRLB model object (\code{models}) is provided, a data frame containing summary information for the models in the model object is returned.
#' 
#' When both an NRLB model object and a model index is provided, a data frame containing the name, footprint size (\code{k}), mode relative affinity
#' (\code{RelativeAffinity}), and the optimal sequence (\code{BestSequence}) is provided for all modes in the model. The nonspecific binding mode is 
#' denoted by \code{NSB} and only has an associated relative affinity. 
#' 
#' @examples
#' model.info()
#' 
#' model.info(nrlb.model = 15)
#' 
#' m = NRLBtools::load.models(fileName = system.file("extdata", "MAX-NRLBConfig.csv", package = "NRLBtools"))
#' model.info(m)
#' 
#' model.info(m, 8)
#'
#' @export
#' 
model.info = function(models = NULL, index = NULL, nrlb.model = NULL) {
  # First see if models AND nrlb.model are null; if so, just print the available models
  if (is.null(models) && is.null(nrlb.model)) {
    return(NRLBModelInfo)
  }
  # Next, handle the case where models is not null BUT index and nlrb.model are all null: display model info
  if (is.null(index) && is.null(nrlb.model)) {
    return(models[[1]])
  }
  # Handle the ambiguous case where fit and nlrb.model is provided
  if (is.null(index) && !is.null(models)) {
    stop("Conflicting arguments: please define either a model or a NRLB model")
  }
  # Now handle the remaining cases, including if an NRLB model is provided
  if (!is.null(nrlb.model)) {
    models = NRLBModels
    index = nrlb.model
  }
  #Get betas and transform them into a matrix
  if (is.null(index)) {
    stop("Model Index Required")
  }
  fit = models[[2]][[index]]
  k = models[[1]]$k[index]
  isMulti = (k=="Multi")
  if (isMulti) {
    output = NULL
    #Loop over all modes
    modes = 1:length(fit)
    for (currMode in modes) {
      if (names(fit)[currMode]=="NSBinding") {
        next
      }
      nuc = fit[[currMode]]$NB
      k = length(nuc)/4
      dim(nuc) = c(4, k)
      if (is.null(fit[[currMode]]$DB)) {
        isDinuc = FALSE
        dinuc = NULL
      } else {
        isDinuc = TRUE
        dinuc = fit[[currMode]]$DB
        dim(dinuc) = c(16, k-1)
        rownames(dinuc) = c("AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT")
      }
      output = rbind(output, .maxSeqHelper(isDinuc, k, nuc, dinuc))
    }
  } else {
    nuc = fit$NB
    k = as.numeric(k)
    dim(nuc) = c(4,k)
    if (is.null(fit$DB)) {
      isDinuc = FALSE
      dinuc = NULL
    } else {
      isDinuc = TRUE
      dinuc = fit$DB
      dim(dinuc) = c(16, k-1)
      rownames(dinuc) = c("AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT")
    }
    output = .maxSeqHelper(isDinuc, k, nuc, dinuc)
  }
  
  # Process depending on the existence of NSB in the model
  opt.aff = output$score
  seqs    = output$seq
  ks      = nchar(output$seq)
  modes   = 1:length(ks)
  if (models[[1]]$NS[index]) {
    nsb     = exp(models[[1]]$NSBind[index])
    opt.aff = c(opt.aff, nsb)
    relaff  = log(opt.aff/max(opt.aff))
    seqs    = c(seqs, "-")
    ks      = c(ks, "-")
    modes   = c(modes, "NSB")
  } else {
    relaff  = log(opt.aff/max(opt.aff))
  }
  # Create output dataframe
  output = data.frame(Mode=modes, k=ks, RelativeAffinity=relaff, BestSequence=seqs, stringsAsFactors = FALSE)
  return(output)
}

.maxSeqHelper = function(isDinuc, k, nuc, dinuc) {
  #Initialize loop (position 1)
  max.list = nuc[,1] #(A C G T)
  char.list= c("A", "C", "G", "T")
  temp.list= char.list
  curr.list = matrix(data=0, 4, 4)
  #Loop over all positions
  for (currPos in 2:k) {
    #Loop over all previous bases
    for (prevBase in 1:4) {
      curr.list[,prevBase] = max.list[prevBase] + nuc[, currPos]
      if (isDinuc) {
        curr.list[,prevBase] = curr.list[,prevBase] + dinuc[((prevBase-1)*4+1):(prevBase*4), (currPos-1)]
      }
    }
    for (currBase in 1:4) {
      max.list[currBase] = max(curr.list[currBase,])
      temp.list[currBase] = paste0(char.list[which.max(curr.list[currBase,])], c("A", "C", "G", "T")[currBase])
    }
    char.list = temp.list
  }
  return(data.frame(score = exp(max(max.list)), 
                    seq   = char.list[which.max(max.list)], 
                    stringsAsFactors = FALSE))
}

.maxseq = function(fits, index, mode) {
  #Get betas and transform them into a matrix
  currfit = fits[[2]][[index]]
  k = fits[[1]]$k[index]
  isMulti = (k=="Multi")
  if (isMulti) {
    currfit = currfit[[mode]]
  }
  nuc = currfit$NB
  k = length(nuc)/4
  dim(nuc) = c(4,k)
  if (is.null(currfit$DB)) {
    isDinuc = FALSE
    dinuc = NULL
  } else {
    isDinuc = TRUE
    dinuc = currfit$DB
    dim(dinuc) = c(16, k-1)
    rownames(dinuc) = c("AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT")
  }
  return(.maxSeqHelper(isDinuc, k, nuc, dinuc))
}