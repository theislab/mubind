#' Generate a PSAM/Energy Logo
#' 
#' @description A function used to visualize models as energy logos. 
#' 
#' @param models an NRLB model object. 
#' @param index numeric indicating the index of the model in the NRLB model object to be visualized. Required when providing a NRLB model object.
#' @param mode numeric indicating the mode of a multi-mode model to be visualized. Required when attempting to visualize a multi-mode model. 
#' @param rc logical indicating that the reverse complement of the model should be displayed.
#' @param betas A \eqn{4l} long numeric vector containing the energy parameters associated with a mononucleotide model with footprint length \eqn{l}.
#' @param title text string to print on top of the energy logo.
#' @param ylim numeric vector of length 2 giving the range of energy values to plot. 
#' @param l.pad numeric indicating the number of empty base pairs to add the left of the energy logo.
#' @param r.pad numeric indicating the number of empty base pairs to add the right of the energy logo.
#' @param l.del numeric indicating the number of base pairs to remove from the left of the energy logo.
#' @param r.del numeric indicating the number of base pairs to remove from the right of the energy logo.
#' @param nrlb.model numeric index of the pre-built NRLB model to visualize. Use \code{model.info} to find the model index. When provided, all other
#' parameters are overriden. 
#' 
#' @return an energy logo image.
#' 
#' @details This function will only plot a single energy logo at a time; when attempting to visualize a multi-mode model, the individual modes of the 
#' model must be visualized separately. When attempting to visualize a model with both mononucelotide and dinucleotide features, the resulting energy
#' logo represents a mononucleotide approximation of the true model. 
#' 
#' \code{betas} can be used to provide explicit values for plotting energy parameters. The numeric vector must provide values for all four bases for
#' every position sequentially. For example, to plot a 3 bp logo, the following vector must be passed: \deqn{{A_1, C_1, G_1, T_1, A_2, C_2, G_2, T_2, 
#' A_3, C_3, G_3, T_3}}
#' 
#' @examples 
#' NRLBtools::logo(nrlb.model = 25)
#' 
#' m = NRLBtools::load.models(fileName = system.file("extdata", "MAX-NRLBConfig.csv", package = "NRLBtools"))
#' NRLBtools::logo(models = m, index = 8)
#'
#' @export
#' 
logo = function(models, index=NULL, mode=NULL, rc=FALSE, betas=NULL, title=NULL, ylim=NULL, l.pad=0, r.pad=0, l.del=0, r.del=0, nrlb.model=NULL) {
  # Handle if an NRLB model is provided
  if (!is.null(nrlb.model)) {
    models = NRLBModels
    index = nrlb.model
    mode = 1
  }
  #Check to see if the input is a core fit
  if (is.null(index)) {
    #Check to see if the input is just a vector of betas
    if (is.null(betas)) {
      stop("Model index needed")
    } else {
      if (rc) {
        fit.output = list(NB=rev(betas))
      } else {
        fit.output = list(NB=betas)
      }
      info.string = "manual-input"
      fit.info = NULL
    }
  } else {
    fit.output = models$Values[[index]]
    fit.info = models$Information[index,]
    info.string = c("k", fit.info$k,"f", fit.info$Flank, 
                    sapply(7:12, function (x) if (as.logical(fit.info[x])) {names(fit.info)[x]} else {NA}))
    info.string = info.string[!is.na(info.string)]
    if (rc) {
      fit.output = .rc(fit.output)
    }
  }
  #Handle Multi-Mode models
  if (class(fit.output[[1]])=="list" || length(fit.output)==9) {
    if (is.null(mode)) {
      stop("Multi-Mode Fit Detected: Mode Index Required for Model")
    } else {
      fit.output = fit.output[[mode]]
      k = length(fit.output$NB)/4
      isMulti = TRUE
      info.string = c("k", fit.info$k,"f", fit.info$Flank, "m", mode, 
                      sapply(7:12, function (x) if (as.logical(fit.info[x])) {names(fit.info)[x]} else {NA}))
      info.string = info.string[!is.na(info.string)]
    }
  } else {
    isMulti = FALSE
  }
  f.name=paste0(paste(info.string,collapse="-"),".eps")
  if (is.null(fit.output$DB)) {
    motif = exp(as.numeric(fit.output$NB))
  } else {
    optimal = .maxseq(models, index, mode)
    rescale = as.numeric(optimal$score)
    top.string = strsplit(as.character(optimal$seq), split="")[[1]]
    PSAM = matrix(data = NA, nrow=4, ncol=length(top.string))
    row.names(PSAM) = c("A", "C", "G", "T")
    
    for (i in 1:length(top.string)) {
      #Loop over all characters at each position
      for (currChar in c("A", "C", "G", "T")) {
        curr.string = top.string
        curr.string[i] = currChar
        curr.string = paste0(curr.string, collapse="")
        #Evaluate model on this
        out = score.seq(sequence = curr.string, models = models, index = index, mode = mode, rescale = rescale)
        #Store
        PSAM[currChar, i] = out[1,1]
      }
    }
    if (rc) {
      motif = rev(as.numeric(PSAM))
    } else {
      motif = as.numeric(PSAM)
    }
  }
  motif = c(rep(0, l.pad*4), motif, rep(0, r.pad*4))
  if (l.del>0) {
    motif = motif[-c(1:(l.del*4))]
  }
  if (r.del>0) {
    motif = motif[-c((length(motif)-r.del*4+1):length(motif))]
  }
  k = length(motif)/4
  dim(motif) = c(4, k)
  motif = log(motif)
  rownames(motif) = c("A", "C", "G", "T")
  motif = apply(motif, 2, function(column) column-mean(column))
  
  colScheme = make_col_scheme(chars=c("A", "C", "G", "T"), cols=c("#5CC93B", "#0D00C4", "#F4B63F", "#BB261A"))
  img = ggseqlogo::ggseqlogo(motif, method='c', font="helvetica_bold", col_scheme=colScheme, coord_cartesian(ylim=20)) + 
    ggplot2::theme_bw() +  
    ggplot2::labs(x = NULL, y = NULL) + 
    ggplot2::ylab(expression(paste(Delta, Delta, "GR/T"))) +
    ggplot2::annotate('rect', xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0, alpha = 0.5, fill='white') +
    ggplot2::scale_y_continuous(breaks=pretty_breaks()) + 
    ggplot2::geom_hline(yintercept=0) +
    ggplot2::theme(text=element_text(size=23), axis.line=element_line(color="black", size=1), 
          axis.ticks=element_line(color="black", size=1), panel.border=element_blank(), 
          panel.grid=element_blank(), aspect.ratio = .5)
  return(img)
}

.rc = function(models, index=NULL) {
  #Dinuc reverse çomplement indicies 
  dinuc.idx = c(16,12,8,4,15,11,7,3,14,10,6,2,13,9,5,1)
  if (is.null(index)) {
    fit.output = models
  } else {
    fit.output = models$Values[[index]]
  }
  #Handle multi-round models
  if (class(fit.output[[1]])=="list" || length(fit.output)==9) {
    #Indices for simple reverse complement
    norm.rev.idx  = c(1, 3, 4, 6, 7, 9)
    #Indices for dinuc reverse complement
    dinuc.rev.idx = c(2, 5, 8)
  } else {
    #Indices for simple reverse complement
    norm.rev.idx  = c(1, 3, 5, 7, 9, 11)
    #Indices for dinuc reverse complement
    dinuc.rev.idx = c(2, 6, 10)
  }
  if (class(fit.output[[1]])=="list") {
    nModes = length(fit.output)
    if ("NSBinding" %in% names(fit.output)) {
      nModes = nModes-1
    }
    for (currMode in 1:nModes) {
      for (i in norm.rev.idx) {
        if (!is.null(fit.output[[currMode]][[i]])) {
          fit.output[[currMode]][[i]] = rev(fit.output[[currMode]][[i]])
        }
      }
      for (i in dinuc.rev.idx) {
        if (!is.null(fit.output[[currMode]][[i]])) {
          temp = matrix(fit.output[[currMode]][[i]], nrow=16)
          temp = temp[dinuc.idx,rev(1:ncol(temp))]
          fit.output[[currMode]][[i]] = as.numeric(temp)
        }
      }
    }
  } else {
    for (i in norm.rev.idx) {
      if (!is.null(fit.output[[i]])) {
        fit.output[[i]] = rev(fit.output[[i]])
      }
    }
    for (i in dinuc.rev.idx) {
      if (!is.null(fit.output[[i]])) {
        temp = matrix(fit.output[[i]], nrow=16)
        temp = temp[dinuc.idx,rev(1:ncol(temp))]
        fit.output[[i]] = as.numeric(temp)
      }
    }
  }
  return(fit.output)
}