package dynamicprogramming;

import base.Array;
import model.Round0Model;
import base.Sequence;

public class FullFeatureNucleotideNoFlank implements DynamicProgramming {
	private boolean isNSBinding;
	private int l;
	private int k;
	private int r0k;
	private int maxFrames;
	private int maxSubFeatures;
	private int totNucFeatures;
	private int	totFeatures;
	private long leftFlank;
	private long leftFlankRC;
	private long rightFlank;
	private long rightFlankRC;
	private long subFeatureMask;
	private double Z;
	private double[] nucAlphas;
	private double[] r0Alphas;
	private double[] nucGradients;
	private double[][] fsInitMatrix;
	private double[][] rsInitMatrix;
	private double[][] hessian;
	
	public FullFeatureNucleotideNoFlank(int l, int k, boolean isNSBinding,
			String lFlank, String rFlank, Round0Model R0Model) {
		this.l 			= l;
		this.k 			= k;
		this.isNSBinding= isNSBinding;
		r0k 			= R0Model.getK();
		maxFrames 		= l-k+1;
		maxSubFeatures	= (int) Math.pow(4, r0k-1);
		totNucFeatures	= 4*k;
		totFeatures		= totNucFeatures;
		r0Alphas		= R0Model.getAlphas();
		leftFlank 		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
		rightFlankRC 	= reverseComplement(leftFlank, lFlank.length());
		rightFlankRC 	= reverse(rightFlankRC, lFlank.length());
		rightFlank 		= (new Sequence(rFlank, 0, rFlank.length())).getValue();
		leftFlankRC 	= reverseComplement(rightFlank, rFlank.length());
		rightFlank 		= reverse(rightFlank, rFlank.length());
		subFeatureMask	= (long) maxSubFeatures - 1;
		
		int fsLFlankInit;
		int rsLFlankInit;	
		if (maxSubFeatures>1) {
			fsLFlankInit = (int) (leftFlank&subFeatureMask);
			rsLFlankInit = (int) (leftFlankRC&subFeatureMask);
		} else {
			fsLFlankInit = 0;
			rsLFlankInit = 0;
		}
		
		//Create initialization matrix
		fsInitMatrix 		= new double[maxFrames][maxSubFeatures];
		rsInitMatrix		= new double[maxFrames][maxSubFeatures];
		for (int i=0; i<maxFrames; i++) {
			fsInitMatrix[i][(int) fsLFlankInit] = 1;
			rsInitMatrix[i][(int) rsLFlankInit] = 1;
		}		
	}
	
	@Override
	public void setAlphas(double[] nucAlphas, double[] dinucAlphas, double[] shapeBetas) {
		this.nucAlphas = nucAlphas;
	}
		
	public double recursiveZ() {
		int startOffset;
		int endOffset;
		int position;
		double fR0Weight	  	= 0;
		double rR0Weight		= 0;
		double fNucWeight	  	= 0;
		double rNucWeight		= 0;
		double[][] fsOffsets	= matrixClone(fsInitMatrix);
		double[][] rsOffsets	= matrixClone(rsInitMatrix);
		double[][] fsWork		= new double[maxFrames][maxSubFeatures];
		double[][] rsWork		= new double[maxFrames][maxSubFeatures];
		long fsRFlank			= rightFlank;
		long rsRFlank			= rightFlankRC;
		long fNewWorkFeat;
		long rNewWorkFeat;
				
		//Loop over variable region
		for (int index=0; index<l; index++) {								//R0, NUC
			startOffset 	 = (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 		 = (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (long currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				for (int newBase=0; newBase<4; newBase++) {
					fNewWorkFeat = (currSubFeat<<2) | newBase;
					rNewWorkFeat = (currSubFeat<<2) | newBase;
					fR0Weight = r0Alphas[(int) fNewWorkFeat];
					rR0Weight = r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
					for (int currOffset=0; currOffset<maxFrames; currOffset++) {
						position = index-currOffset;
						if (currOffset>=startOffset && currOffset<=endOffset) {
							fNucWeight = nucAlphas[position*4 + newBase];
							rNucWeight = nucAlphas[position*4 + newBase];
						} else {
							fNucWeight = 1;
							rNucWeight = 1;
						}
						fsWork[currOffset][(int) (fNewWorkFeat & subFeatureMask)] += 
								fsOffsets[currOffset][(int) currSubFeat]*fR0Weight*fNucWeight;
						rsWork[currOffset][(int) (rNewWorkFeat & subFeatureMask)] += 
									rsOffsets[currOffset][(int) currSubFeat]*rR0Weight*rNucWeight;	
					}
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}
		}
		
		for (int index=0; index<r0k-1; index++) {
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (long currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat = (currSubFeat<<2) | fNewBase;
				rNewWorkFeat = (currSubFeat<<2) | rNewBase;
				fR0Weight = r0Alphas[(int) fNewWorkFeat];
				rR0Weight = r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					fsWork[currOffset][(int) (fNewWorkFeat & subFeatureMask)] +=
							fsOffsets[currOffset][(int) currSubFeat]*fR0Weight;
					rsWork[currOffset][(int) (rNewWorkFeat & subFeatureMask)] +=
							rsOffsets[currOffset][(int) currSubFeat]*rR0Weight;
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}
		}

		Z = 0;
		for (int i=0; i<maxFrames; i++) {
			Z += Array.sum(fsOffsets[i]) + Array.sum(rsOffsets[i]);
		}
		
		fsOffsets	= null;
		rsOffsets	= null;
		fsWork		= null;
		rsWork		= null;
		
		return Z;
	}

	@Override
	public void recursiveGradient() {
		int startOffset;
		int endOffset;
		int nucPosOffset;
		int fNextSubFeat			= 0;
		int rNextSubFeat			= 0;
		double fWeight				= 0;
		double rWeight				= 0;
		double fR0Weight			= 0;
		double rR0Weight			= 0;
		double[][] fsOffsets		= matrixClone(fsInitMatrix);
		double[][] rsOffsets		= matrixClone(rsInitMatrix);
		double[][] fsWork			= new double[maxFrames][maxSubFeatures];
		double[][] rsWork			= new double[maxFrames][maxSubFeatures];
		double[][][] fsGradients	= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsGradients	= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsGradientsWork= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsGradientsWork= new double[totNucFeatures][maxFrames][maxSubFeatures];
		long fNewWorkFeat;
		long rNewWorkFeat;
		long fsRFlank				= rightFlank;
		long rsRFlank				= rightFlankRC;
						
		//Loop over variable region
		for (int index=0; index<l; index++) {					//Loop over all bases
			startOffset 	= (index-k+1 < 0) ? 0 : index-k+1;	//Find the offsets that are currently being considered.
			endOffset 		= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradientsWork[k][i][j] = 0;
						rsGradientsWork[k][i][j] = 0;
					}
				}
			}
			
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				for (int newBase=0; newBase<4; newBase++) {			//Loop over bases to be added.
					fNewWorkFeat= (currSubFeat<<2) | newBase;
					rNewWorkFeat= (currSubFeat<<2) | newBase;
					fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
					rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
					fR0Weight	= r0Alphas[(int) fNewWorkFeat];
					rR0Weight	= r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
					for (int currOffset=0; currOffset<startOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}					
					for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						nucPosOffset= (index-currOffset)*4 + newBase;		//Find alpha that needs to be added.
						fWeight		= nucAlphas[nucPosOffset]*fR0Weight;
						rWeight		= nucAlphas[nucPosOffset]*rR0Weight;
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
						//Lastly, add to gradient features
						fsGradientsWork[nucPosOffset][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsGradientsWork[nucPosOffset][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					}
					for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}
				}				
			}			
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradients[k][i][j] = fsGradientsWork[k][i][j];
						rsGradients[k][i][j] = rsGradientsWork[k][i][j];
					}
				}
			}
		}
		
		for (int index=0; index<r0k-1; index++) {
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradientsWork[k][i][j] = 0;
						rsGradientsWork[k][i][j] = 0;
					}
				}
			}

			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat= (currSubFeat<<2) | fNewBase;
				rNewWorkFeat= (currSubFeat<<2) | rNewBase;
				fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
				fR0Weight	= r0Alphas[(int) fNewWorkFeat];
				rR0Weight	= r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradients[k][i][j] = fsGradientsWork[k][i][j];
						rsGradients[k][i][j] = rsGradientsWork[k][i][j];
					}
				}
			}
		}
		
		Z = 0;
		for (int i=0; i<maxFrames; i++) {
			Z += Array.sum(fsOffsets[i]) + Array.sum(rsOffsets[i]);
		}
		nucGradients = new double[totNucFeatures];
		//Sum over frames to get total gradient
		for (int cgf=0; cgf<totNucFeatures; cgf++) {
			for (int currOffset=0; currOffset<maxFrames; currOffset++) {
				nucGradients[cgf] += Array.sum(fsGradients[cgf][currOffset]) 
						+ Array.sum(rsGradients[cgf][currOffset]);				
			}
		}
		fsOffsets		= null;
		rsOffsets		= null;
		fsWork			= null;
		rsWork			= null;
		fsGradients		= null;
		rsGradients		= null;
		fsGradientsWork	= null;
		rsGradientsWork	= null;
		System.gc();
	}
	
	@Override
	public void recursiveHessian() {
		int startOffset;
		int endOffset;
		int nucPosOffset;
		int fNextSubFeat			= 0;
		int rNextSubFeat			= 0;
		double fWeight				= 0;
		double rWeight				= 0;
		double fR0Weight			= 0;
		double rR0Weight			= 0;
		double fwdUpdate, revUpdate;
		double[][] fsOffsets		= matrixClone(fsInitMatrix);
		double[][] rsOffsets		= matrixClone(rsInitMatrix);
		double[][] fsWork			= new double[maxFrames][maxSubFeatures];
		double[][] rsWork			= new double[maxFrames][maxSubFeatures];
		double[][][] fsGradients	= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsGradients	= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsGradientsWork= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsGradientsWork= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][][] fsHessian	= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] rsHessian	= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] fsHessianWork= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] rsHessianWork= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		long fNewWorkFeat;
		long rNewWorkFeat;
		long fsRFlank				= rightFlank;
		long rsRFlank				= rightFlankRC;
		
		//Loop over variable region
		for (int index=0; index<l; index++) {					//Loop over all bases
			startOffset 	= (index-k+1 < 0) ? 0 : index-k+1;	//Find the offsets that are currently being considered.
			endOffset 		= (index > l-k) ? l-k : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradientsWork[k][i][j] = 0;
						rsGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int l=0; l<totFeatures; l++) {
				for (int k=0; k<totFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsHessianWork[l][k][i][j] = 0;
							rsHessianWork[l][k][i][j] = 0;
						}
					}
				}				
			}
			
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				for (int newBase=0; newBase<4; newBase++) {			//Loop over bases to be added.
					fNewWorkFeat= (currSubFeat<<2) | newBase;
					rNewWorkFeat= (currSubFeat<<2) | newBase;
					fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
					rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
					fR0Weight	= r0Alphas[(int) fNewWorkFeat];
					rR0Weight	= r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
					//Update all function, gradient and hessian sums with R0 Weight only (when not in the active window)
					for (int currOffset=0; currOffset<startOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
							}
						}
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}
					//Update all function, gradient and hessian sums with R0Weight+alpha weight when in active window
					for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						nucPosOffset= (index-currOffset)*4 + newBase;		//Find alpha that needs to be added.
						fWeight		= nucAlphas[nucPosOffset]*fR0Weight;
						rWeight		= nucAlphas[nucPosOffset]*rR0Weight;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rWeight;
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fWeight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rWeight;
							}
						}
						fwdUpdate = fsOffsets[currOffset][currSubFeat]*fWeight;
						revUpdate = rsOffsets[currOffset][currSubFeat]*rWeight;
						fsWork[currOffset][fNextSubFeat]									+= fwdUpdate;
						rsWork[currOffset][rNextSubFeat]									+= revUpdate;
						fsGradientsWork[nucPosOffset][currOffset][fNextSubFeat]				+= fwdUpdate;
						rsGradientsWork[nucPosOffset][currOffset][rNextSubFeat]				+= revUpdate;
						fsHessianWork[nucPosOffset][nucPosOffset][currOffset][fNextSubFeat]	+= fwdUpdate;
						rsHessianWork[nucPosOffset][nucPosOffset][currOffset][rNextSubFeat]	+= revUpdate;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							fwdUpdate = fsGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							fsHessianWork[nucPosOffset][cgf][currOffset][fNextSubFeat] += fwdUpdate;
							rsHessianWork[nucPosOffset][cgf][currOffset][rNextSubFeat] += revUpdate;
						}
					}
					//Update all function, gradient and hessian sums with R0 Weight only (when not in the active window)
					for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
							}
						}
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}
				}				
			}			
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradients[k][i][j] = fsGradientsWork[k][i][j];
						rsGradients[k][i][j] = rsGradientsWork[k][i][j];
					}
				}
			}
			for (int l=0; l<totFeatures; l++) {
				for (int k=0; k<totFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsHessian[l][k][i][j] = fsHessianWork[l][k][i][j];
							rsHessian[l][k][i][j] = rsHessianWork[l][k][i][j];
						}
					}
				}				
			}
		}
		
		for (int index=0; index<r0k-1; index++) {
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradientsWork[k][i][j] = 0;
						rsGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int l=0; l<totFeatures; l++) {
				for (int k=0; k<totFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsHessianWork[l][k][i][j] = 0;
							rsHessianWork[l][k][i][j] = 0;
						}
					}
				}				
			}

			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat= (currSubFeat<<2) | fNewBase;
				rNewWorkFeat= (currSubFeat<<2) | rNewBase;
				fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
				fR0Weight	= r0Alphas[(int) fNewWorkFeat];
				rR0Weight	= r0Alphas[(int) reverseComplement(rNewWorkFeat, r0k)];
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totFeatures; cgf++) {
						fsGradientsWork[cgf][currOffset][fNextSubFeat] += fsGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsGradientsWork[cgf][currOffset][rNextSubFeat] += rsGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
						}
					}
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}				
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsGradients[k][i][j] = fsGradientsWork[k][i][j];
						rsGradients[k][i][j] = rsGradientsWork[k][i][j];
					}
				}
			}
			for (int l=0; l<totFeatures; l++) {
				for (int k=0; k<totFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsHessian[l][k][i][j] = fsHessianWork[l][k][i][j];
							rsHessian[l][k][i][j] = rsHessianWork[l][k][i][j];
						}
					}
				}				
			}
		}
		
		Z = 0;
		for (int i=0; i<maxFrames; i++) {
			Z += Array.sum(fsOffsets[i]) + Array.sum(rsOffsets[i]);
		}
		nucGradients= new double[totNucFeatures];
		hessian		= (isNSBinding) ? new double[totFeatures+1][totFeatures+1] : new double[totFeatures][totFeatures];
		//Sum over frames to get total gradient and hessian
		for (int cgf=0; cgf<totFeatures; cgf++) {
			for (int currOffset=0; currOffset<maxFrames; currOffset++) {
				nucGradients[cgf] += Array.sum(fsGradients[cgf][currOffset]) 
						+ Array.sum(rsGradients[cgf][currOffset]);				
			}
			for (int cgf2=0; cgf2<=cgf; cgf2++) {
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					hessian[cgf][cgf2] += Array.sum(fsHessian[cgf][cgf2][currOffset]) 
							+ Array.sum(rsHessian[cgf][cgf2][currOffset]);
				}
			}
		}
		//Symmetrize
		for (int cgf=0; cgf<totFeatures; cgf++) {
			for (int cgf2=cgf+1; cgf2<totFeatures; cgf2++) {
				hessian[cgf][cgf2] = hessian[cgf2][cgf];
			}
		}
		
		fsOffsets			= null;
		rsOffsets			= null;
		fsWork				= null;
		rsWork				= null;
		fsGradients			= null;
		rsGradients			= null;
		fsGradientsWork		= null;
		rsGradientsWork		= null;
		fsHessian			= null;
		rsHessian			= null;
		fsHessianWork		= null;
		rsHessianWork		= null;
		System.gc();
	}

	@Override
	public double getZ() {
		return Z;
	}

	@Override
	public double[] getNucGradients() {
		return Array.clone(nucGradients);
	}

	@Override
	public double[] getDinucGradients() {
		return null;
	}

	@Override
	public double[] getShapeGradients() {
		return null;
	}
	
	@Override
	public double[][] getHessian() {
		return matrixClone(hessian);
	}
}
