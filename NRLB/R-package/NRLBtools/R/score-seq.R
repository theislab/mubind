#' Score sequences with a model
#' 
#' @description A function that computes relative affinities at every possible offset within a DNA sequence. It return either the raw values or display them
#' in a plot.
#' 
#' @param sequence a text string containing the DNA sequence to be evaluated.
#' @param models an NRLB model object.
#' @param index numeric indicating the index of the model in the NRLB model object to be used for scoring. Required when providing a NRLB model object.
#' @param mode numeric indicating the mode of a multi-mode model to be used for scoring. Required when attempting to score with a multi-mode model.
#' @param rc logical indicating that the reverse complement of the model should be used when scoring the sequence.
#' @param rescale a numeric value to divide all displayed or output relative affinities by. Default value is 1.
#' @param plot logical indicating whether the affinity scores along the sequence should be visualized in a plot (\code{true}) or output as a numeric matrix
#' (\code{false}).
#' @param nPeaks if \code{plot=T}, the number of highest-affinity peaks to identify within the sequence; prints the rescaled relative affinity, relative
#' position, and sequence of the peaks in descending order of affinity.
#' @param annotate if \code{plot=T} and \code{nPeaks>0}, numbers \code{nPeaks} within the affinity plot in descending order of relative affinity.
#' @param nrlb.model numeric index of the pre-built NRLB model to visualize. Use \code{model.info} to find the model index. When provided, all other
#' model input and selection arguments are overriden.
#' 
#' @return if \code{plot=F}, a numeric matrix of relative affinities, if \code{plot=T}, a ggplot2 plot object. 
#' 
#' @details For an input sequence of length \eqn{l}, the selected model is evaluated at \eqn{l-k+1} positions on both the forward and reverse strands. Here, k 
#' refers to the footprint size of the model. The position value in the plots and output matrix corresponds to the leftmost base in the model. For example, 
#' if you have a 4 bp long model evaluated on a 6 bp long sequence ACGTTG, the model will be evaluated at 3 positions on the forward strand: 
#' \tabular{cc}{
#' Position \tab Sequence \cr
#' 1 \tab ACGT \cr
#' 2 \tab CGTT \cr
#' 3 \tab GTTG \cr
#' }
#' 
#' @examples
#'
#' @export
#' 
score.seq = function(sequence, models, index=NULL, mode=NULL, rc=FALSE, rescale=1, plot=FALSE, nPeaks=NULL, annotate=FALSE, nrlb.model=NULL) {
  dnastr.seq = Biostrings::DNAString(sequence)
  # See if a pre-built model should be used 
  if (!is.null(nrlb.model)) {
    models = NRLBModels
    index = nrlb.model
    mode = 1
  }
  # Isolate fit
  if (is.null(models)) {
    stop("Models required")
  }
  if (is.null(index)) {
    stop("Index required")
  }
  fit = models[[2]][[index]]
  #Check to see if input is a multi-mode model
  if (models[[1]]$k[index]=="Multi") {
    if (is.null(mode)) {
      stop("Multi-Mode Fit Detected: Mode Index Required")
    } else {
      fit = fit[[mode]]
      k = length(fit$NB)/4
    }
  } else {
    k = as.numeric(models[[1]]$k[index])
  }
  adjK = k-1
  nuc = fit$NB
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
  
  # Convert sequence to numeric for rapid indexing
  seq.string = deparse(substitute(sequence))
  if (!is.null(nrlb.model)) {
    fit.string = NRLBModelInfo$Protein[index]
  } else {
    fit.string = deparse(substitute(models))
  }
  fSeq = abs(Biostrings::toComplex(dnastr.seq, c(A=1, C=2, G=3, T=4, N=0)))
  rSeq = abs(Biostrings::toComplex(Biostrings::reverseComplement(dnastr.seq), c(A=1, C=2, G=3, T=4, N=0)))
  charFSeq = sequence
  charRSeq = as.character(Biostrings::reverseComplement(dnastr.seq))
  l = nchar(sequence)
  adjL = l+1
  
  # Score
  score = sapply(1:(l-adjK), FUN=function(x) .fastScore(substr(charFSeq, x, x+adjK), substr(charRSeq, adjL-x-adjK, adjL-x), 
                                                        fSeq[x:(x+adjK)], rSeq[(adjL-x-adjK):(adjL-x)], k, isDinuc, nuc, dinuc))
  score = score/.maxseq(models, index, mode)$score
  score = score/rescale
  # RC if needed
  if(rc) {
    score = score[c(2, 1),]
  }
  
  # Plot?
  if (!plot) {
    return(score)
  }
  max.score = max(score)
  
  # Create a proper scale for the image
  multiplier = 1
  while(max.score*multiplier<10) {
    multiplier = multiplier*10
  }
  # Round up to nearest integer
  upper.bound = ceiling(max.score*multiplier)
  # Find optimal step size
  divs = signif(seq(-upper.bound, upper.bound, length.out = 11)/multiplier, digits=2)
  
  # Find top sites and rank them
  if (!is.null(nPeaks)) {
    idx = c(score[1,], score[2,])
    idx = cbind(idx, c(1:ncol(score), -(1:ncol(score))))
    idx = idx[order(-idx[,1]),]
    if (nPeaks>nrow(idx)) {
      nPeaks = nrow(idx)
    }
    Sequence = character(nPeaks)
    for (i in 1:nPeaks) {
      Sequence[i] = as.character(dnastr.seq[abs(idx[i,2]):(abs(idx[i,2])+k-1)])
    }
    peaks = data.frame(Affinity=idx[1:nPeaks,1], Position=idx[1:nPeaks,2], Sequence)
    cat(paste0(fit.string," scores in ",seq.string,"\n"))
    print(peaks)
  }
  
  # Plot
  df = data.frame(Group=as.character(rep(1:2,each=ncol(score))), Position=c(1:ncol(score),1:ncol(score)), 
                  Affinity=c(score[1,], -score[2,]))

  p = ggplot(df, aes(x=Position, y=Affinity, colour=Group)) +
    theme_bw() +
    geom_line() +
    ylab(bquote(Relative~~Affinity~~(10^{.(log10(10/multiplier))}))) + 
    coord_fixed(ylim=c(-max.score,max.score)) +
    scale_y_continuous(breaks=divs, labels=format(abs(divs*multiplier/10), nsmall=2)) +
    theme(aspect.ratio=1, text=element_text(size=17, family="Helvetica")) +
    theme(legend.title=element_blank(), legend.position=c(0.01,0.01), legend.justification=c(0,0)) +
    theme(axis.title.x = element_text(vjust=0), axis.title.y = element_text(vjust=0)) +
    labs(title = paste0(fit.string," scores in ",seq.string))
    p = p+scale_color_manual(values=c("#000000", "#FF0000"), labels=c("Forward", "Reverse"))
  if (annotate && !is.null(nPeaks)) {
    xpos = abs(peaks$Position)+ncol(score)*.015
    ypos = sign(peaks$Position)*peaks$Affinity+max.score*.015
    p = p + annotate("text", x=xpos, y=ypos, label=as.character(1:nPeaks))
  }
  return(p)
}


#fast sequence scorer, optimized to work with score.genome
.fastScore = function(charFSeq, charRSeq, fSeq, rSeq, k, isDinuc, nuc, dinuc) {
  if (all(fSeq!=0)) {
    fTotal = 0
    rTotal = 0
    if (isDinuc) {
      for (j in 1:k) {
        fTotal = fTotal+nuc[fSeq[j],j]
        rTotal = rTotal+nuc[rSeq[j],j]
        if (j<k) {
          fTotal = fTotal+as.numeric(dinuc[substr(charFSeq, j, j+1),j])
          rTotal = rTotal+as.numeric(dinuc[substr(charRSeq, j, j+1),j])
        }
      }
    } else {
      for (j in 1:k) {
        fTotal = fTotal+nuc[fSeq[j],j]
        rTotal = rTotal+nuc[rSeq[j],j]
      }
    }
    output = c(exp(fTotal), exp(rTotal))
    return(output)
  } else {
    return(c(NA, NA))
  }
}
