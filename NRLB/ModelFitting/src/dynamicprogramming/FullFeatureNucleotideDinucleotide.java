package dynamicprogramming;

import base.Array;
import model.Round0Model;
import base.Sequence;

public class FullFeatureNucleotideDinucleotide implements DynamicProgramming {
	private boolean isNSBinding;
	private int l;
	private int k;
	private int flankLength;
	private int r0k;
	private int maxFrames;
	private int maxSubFeatures;
	private int totNucFeatures;
	private int totDinucFeatures;
	private int totFeatures;
	private long leftFlank;
	private long leftFlankRC;
	private long rightFlank;
	private long rightFlankRC;
	private long subFeatureMask;
	private long r0FeatureMask;
	private double Z;
	private double[] nucAlphas;
	private double[] nucGradients;
	private double[] dinucAlphas;
	private double[] dinucGradients;
	private double[] r0Alphas;
	private double[][] fsInitMatrix;
	private double[][] rsInitMatrix;
	private double[][] hessian;
	
	public FullFeatureNucleotideDinucleotide(int l, int k, boolean isNSBinding, 
			int flankLength, String lFlank, String rFlank, Round0Model R0Model) {
		this.l 			= l;
		this.k 			= k;
		this.isNSBinding= isNSBinding;
		this.flankLength=flankLength;
		r0k 			= R0Model.getK();
		maxFrames 		= l-k+1+2*flankLength;
		maxSubFeatures	= (int) Math.max(4, Math.pow(4, r0k-1));
		totNucFeatures	= 4*k;
		totDinucFeatures= 16*(k-1);
		totFeatures		= totNucFeatures + totDinucFeatures;
		r0Alphas		= R0Model.getAlphas();
		leftFlank 		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
		rightFlankRC 	= reverseComplement(leftFlank, lFlank.length());
		rightFlankRC 	= reverse(rightFlankRC, lFlank.length());
		rightFlank 		= (new Sequence(rFlank, 0, rFlank.length())).getValue();
		leftFlankRC 	= reverseComplement(rightFlank, rFlank.length());
		rightFlank 		= reverse(rightFlank, rFlank.length());
		subFeatureMask	= (long) maxSubFeatures - 1;
		r0FeatureMask	= (long) Math.pow(4, r0k) - 1;				//Mask to select the Round0 Feature
		long flankMask	= (long) Math.pow(4, flankLength) - 1;
		
		int fsLFlankInit;
		int rsLFlankInit;	
		if (r0k-1>flankLength) {
			fsLFlankInit = (int) (((leftFlank >> 2*flankLength)) & (long) ((Math.pow(4, r0k-1-flankLength)) - 1));
			rsLFlankInit = (int) ((leftFlankRC >> 2*flankLength) & (long) ((Math.pow(4, r0k-1-flankLength)) - 1));
		} else {
			fsLFlankInit = (int) ((leftFlank >> 2*flankLength)) & 3;
			rsLFlankInit = (int) ((leftFlankRC >> 2*flankLength) & 3);
		}
		
		leftFlank		= leftFlank & flankMask;
		leftFlank		= reverse(leftFlank, flankLength);
		leftFlankRC		= leftFlankRC & flankMask;
		leftFlankRC		= reverse(leftFlankRC, flankLength);
		
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
		this.nucAlphas	= nucAlphas;
		this.dinucAlphas= dinucAlphas;
	}
	
	public double recursiveZ() {
		int startOffset;
		int endOffset;
		int position;
		double fR0Weight		= 0;
		double rR0Weight		= 0;
		double fNucWeight		= 0;
		double rNucWeight		= 0;
		double fDinucWeight		= 0;
		double rDinucWeight		= 0;
		double[][] fsOffsets 	= matrixClone(fsInitMatrix);
		double[][] rsOffsets	= matrixClone(rsInitMatrix);
		double[][] fsWork		= new double[maxFrames][maxSubFeatures];
		double[][] rsWork		= new double[maxFrames][maxSubFeatures];
		long fsLFlank			= leftFlank;
		long fsRFlank			= rightFlank;
		long rsLFlank			= leftFlankRC;
		long rsRFlank			= rightFlankRC;
		long fNewWorkFeat;
		long rNewWorkFeat;

		//Loop over left fixed region
		for (int index=0; index<flankLength; index++) {									//NO ROUND0 MODEL HERE
				startOffset 	 = (index-k+1 < 0) ? 0 : index-k+1;
				endOffset 		 = (index > maxFrames-1) ? maxFrames-1 : index;
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsWork[i][j] = 0;
						rsWork[i][j] = 0;
					}
				}
				//Find new base
				int fNewBase = (int) fsLFlank & 3;
				int rNewBase = (int) rsLFlank & 3;
				fsLFlank >>= 2;
				rsLFlank >>= 2;
			for (long currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat = (currSubFeat<<2) | fNewBase;
				rNewWorkFeat = (currSubFeat<<2) | rNewBase;
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					position = index-currOffset;
					if (currOffset>=startOffset && currOffset<=endOffset) {
						if (position < 1) {
							fDinucWeight = 1;
							rDinucWeight = 1;
						} else {
							fDinucWeight = dinucAlphas[(position-1)*16 + (int) (fNewWorkFeat & 15)];
							rDinucWeight = dinucAlphas[(position-1)*16 + (int) (rNewWorkFeat & 15)];
						}
						fNucWeight = nucAlphas[position*4 + fNewBase];
						rNucWeight = nucAlphas[position*4 + rNewBase];
					} else {
						fDinucWeight = 1;
						rDinucWeight = 1;
						fNucWeight = 1;
						rNucWeight = 1;
					}
					fsWork[currOffset][(int) (fNewWorkFeat & subFeatureMask)] += 
							fsOffsets[currOffset][(int) currSubFeat]*fNucWeight*fDinucWeight;
					rsWork[currOffset][(int) (rNewWorkFeat & subFeatureMask)] +=
							rsOffsets[currOffset][(int) currSubFeat]*rNucWeight*rDinucWeight;
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}
		}

		//Loop over variable region
		for (int index=flankLength; index<l+flankLength; index++) {								//R0, NUC, DINUC
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
					fR0Weight = r0Alphas[(int) (fNewWorkFeat & r0FeatureMask)];
					rR0Weight = r0Alphas[(int) reverseComplement((rNewWorkFeat & r0FeatureMask), r0k)];
					for (int currOffset=0; currOffset<maxFrames; currOffset++) {
						position = index-currOffset;
						if (currOffset>=startOffset && currOffset<=endOffset) {
							if (position < 1) {
								fDinucWeight = 1;
								rDinucWeight = 1;
							} else {
								fDinucWeight = dinucAlphas[(position-1)*16 + (int) (fNewWorkFeat & 15)];
								rDinucWeight = dinucAlphas[(position-1)*16 + (int) (rNewWorkFeat & 15)];
							}
							fNucWeight = nucAlphas[position*4 + newBase];
							rNucWeight = nucAlphas[position*4 + newBase];
						} else {
							fDinucWeight = 1;
							rDinucWeight = 1;
							fNucWeight = 1;
							rNucWeight = 1;
						}
						fsWork[currOffset][(int) (fNewWorkFeat & subFeatureMask)] += 
								fsOffsets[currOffset][(int) currSubFeat]*fR0Weight*fNucWeight*fDinucWeight;
						rsWork[currOffset][(int) (rNewWorkFeat & subFeatureMask)] += 
								rsOffsets[currOffset][(int) currSubFeat]*rR0Weight*rNucWeight*rDinucWeight;
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

		//Loop over right fixed region
		for (int index=l+flankLength; index<l+2*flankLength; index++) {								//R0, NUC, DINUC
			startOffset 	 = (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 		 = (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			//Find new base
			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (long currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat = (currSubFeat<<2) | fNewBase;
				rNewWorkFeat = (currSubFeat<<2) | rNewBase;
				if (index >= l+flankLength+r0k-1) {
					fR0Weight = 1;
					rR0Weight = 1;
				} else {
					fR0Weight = r0Alphas[(int) (fNewWorkFeat & r0FeatureMask)];
					rR0Weight = r0Alphas[(int) reverseComplement((rNewWorkFeat & r0FeatureMask), r0k)];
				}
				for (int currOffset=0; currOffset<maxFrames; currOffset++) {
					position = index-currOffset;
					if (currOffset>=startOffset && currOffset<=endOffset) {
						if (position < 1) {
							fDinucWeight = 1;
							rDinucWeight = 1;
						} else {
							fDinucWeight = dinucAlphas[(position-1)*16 + (int) (fNewWorkFeat & 15)];
							rDinucWeight = dinucAlphas[(position-1)*16 + (int) (rNewWorkFeat & 15)];
						}
						fNucWeight = nucAlphas[position*4 + fNewBase];
						rNucWeight = nucAlphas[position*4 + rNewBase];
					} else {
						fDinucWeight = 1;
						rDinucWeight = 1;
						fNucWeight = 1;
						rNucWeight = 1;
					}
					fsWork[currOffset][(int) (fNewWorkFeat & subFeatureMask)] += 
						fsOffsets[currOffset][(int) currSubFeat]*fR0Weight*fNucWeight*fDinucWeight;
					rsWork[currOffset][(int) (rNewWorkFeat & subFeatureMask)] +=
						rsOffsets[currOffset][(int) currSubFeat]*rR0Weight*rNucWeight*rDinucWeight;
				}
			}
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsOffsets[i][j] = fsWork[i][j];
					rsOffsets[i][j] = rsWork[i][j];
				}
			}
		}
		
		if (r0k-1 > flankLength) {
			for (int index=0; index<r0k-1-flankLength; index++) {
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
					fR0Weight = r0Alphas[(int) (fNewWorkFeat & r0FeatureMask)];
					rR0Weight = r0Alphas[(int) reverseComplement((rNewWorkFeat & r0FeatureMask), r0k)];
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
		int position;
		int nucPosOffset;
		int fDinucPosOffset;
		int rDinucPosOffset;
		int fNextSubFeat;
		int rNextSubFeat;
		double fWeight					= 0;
		double rWeight					= 0;
		double fR0Weight				= 0;
		double rR0Weight				= 0;
		double[][] fsOffsets			= matrixClone(fsInitMatrix);
		double[][] rsOffsets			= matrixClone(rsInitMatrix);
		double[][] fsWork				= new double[maxFrames][maxSubFeatures];
		double[][] rsWork				= new double[maxFrames][maxSubFeatures];
		double[][][] fsNucGradients		= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsNucGradients		= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsDinucGradients	= new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsDinucGradients	= new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsNucGradientsWork	= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsNucGradientsWork = new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsDinucGradientsWork=new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsDinucGradientsWork=new double[totDinucFeatures][maxFrames][maxSubFeatures];
		long fNewWorkFeat;	
		long rNewWorkFeat;
		long fsLFlank					= leftFlank;
		long fsRFlank					= rightFlank;
		long rsLFlank					= leftFlankRC;
		long rsRFlank					= rightFlankRC;
		
		//Loop over left fixed region
		for (int index=0; index<flankLength; index++) {							//NO ROUND0 MODEL HERE
			startOffset			= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
					}
				}
			}
			
			//Find new base
			int fNewBase = (int) fsLFlank & 3;
			int rNewBase = (int) rsLFlank & 3;
			fsLFlank >>= 2;
			rsLFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat = (currSubFeat<<2) | fNewBase;
				rNewWorkFeat = (currSubFeat<<2) | rNewBase;
				fNextSubFeat = (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat = (int) (rNewWorkFeat & subFeatureMask);			
				for (int currOffset=0; currOffset<startOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat];
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat];
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat];
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat];
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat];
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat];
				}
				for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					position		= index-currOffset;
					nucPosOffset	= position*4;
					fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
					rDinucPosOffset = (position-1)*16 + (int) (rNewWorkFeat & 15);
					fWeight			= nucAlphas[nucPosOffset + fNewBase];
					rWeight			= nucAlphas[nucPosOffset + rNewBase];
					if (position >= 1) {
						fWeight *= dinucAlphas[fDinucPosOffset];
						rWeight *= dinucAlphas[rDinucPosOffset];
					}
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					//Lastly, add to gradient features
					fsNucGradientsWork[nucPosOffset + fNewBase][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;						
					rsNucGradientsWork[nucPosOffset + rNewBase][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					if (position>=1) {
						fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					}
				}
				for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat];
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat];
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat];
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat];
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat];
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat];
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
					}
				}
			}
		}
		
		//Loop over variable region
		for (int index=flankLength; index<l+flankLength; index++) {				//Loop over all bases
			startOffset 		= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
					}
				}
			}
			
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				for (int newBase=0; newBase<4; newBase++) {			//Loop over bases to be added.
					fNewWorkFeat= (currSubFeat<<2) | newBase;
					rNewWorkFeat= (currSubFeat<<2) | newBase;
					fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
					rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);			
					fR0Weight	= r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight	= r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
					
					for (int currOffset=0; currOffset<startOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}
					for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						position		= index-currOffset;				//Find alpha that needs to be added.
						nucPosOffset	= position*4 + newBase;
						fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
						rDinucPosOffset	= (position-1)*16 + (int) (rNewWorkFeat & 15);
						fWeight			= nucAlphas[nucPosOffset]*fR0Weight;
						rWeight			= nucAlphas[nucPosOffset]*rR0Weight;
						if (position >= 1) {
							fWeight *= dinucAlphas[fDinucPosOffset];
							rWeight *= dinucAlphas[rDinucPosOffset];
						}
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
						//Lastly, add to gradient features
						fsNucGradientsWork[nucPosOffset][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsNucGradientsWork[nucPosOffset][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
						if (position>=1) {
							fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
							rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
						}
					}
					for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						//First, update gradient features
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
					}
				}
			}
		}
		
		//Loop over right fixed region
		for (int index=l+flankLength; index<l+2*flankLength; index++) {								//R0, NUC, DINUC
			startOffset 		= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
					}
				}
			}
			
			//Find new base
			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat= (currSubFeat<<2) | fNewBase;
				rNewWorkFeat= (currSubFeat<<2) | rNewBase;
				fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
				fR0Weight	= 1;
				rR0Weight	= 1;
				if (index < l+flankLength+r0k-1) {
					fR0Weight = r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight = r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
				}

				for (int currOffset=0; currOffset<startOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
				}
				for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					position		= index-currOffset;
					nucPosOffset	= position*4;
					fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
					rDinucPosOffset	= (position-1)*16 + (int) (rNewWorkFeat & 15);
					fWeight			= nucAlphas[nucPosOffset + fNewBase]*fR0Weight;
					rWeight			= nucAlphas[nucPosOffset + rNewBase]*rR0Weight;
					if (position >= 1) {
						fWeight *= dinucAlphas[fDinucPosOffset];
						rWeight *= dinucAlphas[rDinucPosOffset];
					}
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					//Lastly, add to gradient features
					fsNucGradientsWork[nucPosOffset + fNewBase][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;						
					rsNucGradientsWork[nucPosOffset + rNewBase][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					if (position>=1) {
						fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fWeight;
						rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rWeight;
					}
				}
				for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					//First, update gradient features
					for (int cgf=0; cgf<totNucFeatures; cgf++) {
						fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
					}
					for (int cgf=0; cgf<totDinucFeatures; cgf++) {
						fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
						rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
					}
				}
			}
		}
		
		if (r0k-1 > flankLength) {
			for (int index=0; index<r0k-1-flankLength; index++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsWork[i][j] = 0;
						rsWork[i][j] = 0;
					}
				}
				for (int k=0; k<totNucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsNucGradientsWork[k][i][j] = 0;
							rsNucGradientsWork[k][i][j] = 0;
						}
					}
				}
				for (int k=0; k<totDinucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							rsDinucGradientsWork[k][i][j] = 0;
							fsDinucGradientsWork[k][i][j] = 0;
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
					fR0Weight	= r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight	= r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
					for (int currOffset=0; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
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
							fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
							rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
						}
					}
				}
				for (int k=0; k<totDinucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
							rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
						}
					}
				}
			}			
		}
		
		Z = 0;
		for (int i=0; i<maxFrames; i++) {
			Z += Array.sum(fsOffsets[i]) + Array.sum(rsOffsets[i]);
		}
		//Sum over frames to get total gradient
		nucGradients = new double[totNucFeatures];
		for (int cgf=0; cgf<totNucFeatures; cgf++) {
			for (int currOffset=0; currOffset<maxFrames; currOffset++) {
				nucGradients[cgf] += Array.sum(fsNucGradients[cgf][currOffset]) 
						+ Array.sum(rsNucGradients[cgf][currOffset]);				
			}
		}
		dinucGradients = new double[totDinucFeatures];
		for (int cgf=0; cgf<totDinucFeatures; cgf++) {
			for (int currOffset=0; currOffset<maxFrames; currOffset++) {
				dinucGradients[cgf] += Array.sum(fsDinucGradients[cgf][currOffset]) 
						+ Array.sum(rsDinucGradients[cgf][currOffset]);				
			}
		}
		fsOffsets			= null;
		rsOffsets			= null;
		fsWork				= null;
		rsWork				= null;
		fsNucGradients		= null;
		rsNucGradients		= null;
		fsDinucGradients	= null;
		rsDinucGradients	= null;
		fsNucGradientsWork	= null;
		rsNucGradientsWork	= null;
		fsDinucGradientsWork= null;
		rsDinucGradientsWork= null;
		System.gc();
	}
	
	@Override
	public void recursiveHessian() {
		int startOffset;
		int endOffset;
		int position;
		int nucPosOffset, fNucPosOffset, rNucPosOffset;;
		int fDinucPosOffset;
		int rDinucPosOffset;
		int fNextSubFeat;
		int rNextSubFeat;
		int tempIdx;
		double fwdUpdate, revUpdate;
		double fWeight					= 0;
		double rWeight					= 0;
		double fR0Weight				= 0;
		double rR0Weight				= 0;
		double[][] fsOffsets			= matrixClone(fsInitMatrix);
		double[][] rsOffsets			= matrixClone(rsInitMatrix);
		double[][] fsWork				= new double[maxFrames][maxSubFeatures];
		double[][] rsWork				= new double[maxFrames][maxSubFeatures];
		double[][][] fsNucGradients		= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsNucGradients		= new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsDinucGradients	= new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsDinucGradients	= new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsNucGradientsWork = new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsNucGradientsWork = new double[totNucFeatures][maxFrames][maxSubFeatures];
		double[][][] fsDinucGradientsWork=new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][] rsDinucGradientsWork=new double[totDinucFeatures][maxFrames][maxSubFeatures];
		double[][][][] fsHessian		= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] rsHessian		= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] fsHessianWork	= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		double[][][][] rsHessianWork	= new double[totFeatures][totFeatures][maxFrames][maxSubFeatures];
		long fNewWorkFeat;	
		long rNewWorkFeat;
		long fsLFlank					= leftFlank;
		long fsRFlank					= rightFlank;
		long rsLFlank					= leftFlankRC;
		long rsRFlank					= rightFlankRC;
		
		//Loop over left fixed region
		for (int index=0; index<flankLength; index++) {							//NO ROUND0 MODEL HERE
			startOffset			= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
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
			
			//Find new base
			int fNewBase = (int) fsLFlank & 3;
			int rNewBase = (int) rsLFlank & 3;
			fsLFlank >>= 2;
			rsLFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat = (currSubFeat<<2) | fNewBase;
				rNewWorkFeat = (currSubFeat<<2) | rNewBase;
				fNextSubFeat = (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat = (int) (rNewWorkFeat & subFeatureMask);			
				for (int currOffset=0; currOffset<startOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat];
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat];
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat];
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat];
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat];
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat];
						}
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat];
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat];
				}
				for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					position		= index-currOffset;
					fNucPosOffset	= position*4 + fNewBase;
					rNucPosOffset	= position*4 + rNewBase;
					fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
					rDinucPosOffset = (position-1)*16 + (int) (rNewWorkFeat & 15);
					fWeight			= nucAlphas[fNucPosOffset];
					rWeight			= nucAlphas[rNucPosOffset];
					if (position >= 1) {
						fWeight *= dinucAlphas[fDinucPosOffset];
						rWeight *= dinucAlphas[rDinucPosOffset];
					}
					//First, update gradient features
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fWeight;
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rWeight;
						}
					}
					fwdUpdate = fsOffsets[currOffset][currSubFeat]*fWeight;
					revUpdate = rsOffsets[currOffset][currSubFeat]*rWeight;
					fsWork[currOffset][fNextSubFeat]									+= fwdUpdate;
					rsWork[currOffset][rNextSubFeat]									+= revUpdate;
					fsNucGradientsWork[fNucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
					rsNucGradientsWork[rNucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
					fsHessianWork[fNucPosOffset][fNucPosOffset][currOffset][fNextSubFeat] += fwdUpdate;
					rsHessianWork[rNucPosOffset][rNucPosOffset][currOffset][rNextSubFeat] += revUpdate;
					
					if (position>=1) {
						fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat]				+= fwdUpdate;
						rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat]				+= revUpdate;
						fDinucPosOffset += 4*k;
						rDinucPosOffset += 4*k;
						fsHessianWork[fDinucPosOffset][fDinucPosOffset][currOffset][fNextSubFeat]	+= fwdUpdate;
						rsHessianWork[rDinucPosOffset][rDinucPosOffset][currOffset][rNextSubFeat]	+= revUpdate;
						fsHessianWork[fDinucPosOffset][fNucPosOffset][currOffset][fNextSubFeat]		+= fwdUpdate;
						rsHessianWork[rDinucPosOffset][rNucPosOffset][currOffset][rNextSubFeat]		+= revUpdate;
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							fsHessianWork[fNucPosOffset][cgf][currOffset][fNextSubFeat]				+= fwdUpdate;
							rsHessianWork[rNucPosOffset][cgf][currOffset][rNextSubFeat]				+= revUpdate;
							fsHessianWork[fDinucPosOffset][cgf][currOffset][fNextSubFeat]			+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][cgf][currOffset][rNextSubFeat]			+= revUpdate;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							tempIdx = cgf+4*k;
							fsHessianWork[tempIdx][fNucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
							rsHessianWork[tempIdx][rNucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
							fsHessianWork[fDinucPosOffset][tempIdx][currOffset][fNextSubFeat]		+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][tempIdx][currOffset][rNextSubFeat]		+= revUpdate;
						}
					} else {
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							fsHessianWork[fNucPosOffset][cgf][currOffset][fNextSubFeat] += fwdUpdate;
							rsHessianWork[rNucPosOffset][cgf][currOffset][rNextSubFeat] += revUpdate;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							tempIdx = cgf+4*k;
							fsHessianWork[fNucPosOffset][tempIdx][currOffset][fNextSubFeat] += fwdUpdate;
							rsHessianWork[rNucPosOffset][tempIdx][currOffset][rNextSubFeat] += revUpdate;
						}	
					}
				}
				for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat];
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat];
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat];
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat];
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat];
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat];
						}
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat];
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat];
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
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
		
		//Loop over variable region
		for (int index=flankLength; index<l+flankLength; index++) {				//Loop over all bases
			startOffset 		= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
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
					fR0Weight	= r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight	= r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
					
					for (int currOffset=0; currOffset<startOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							if (cgf<totNucFeatures) {
								fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							if (cgf<totDinucFeatures) {
								fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
							}
						}
						//Next, update non-gradient sums
						fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
						rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
					}
					for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						position		= index-currOffset;				//Find alpha that needs to be added.
						nucPosOffset	= position*4 + newBase;
						fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
						rDinucPosOffset	= (position-1)*16 + (int) (rNewWorkFeat & 15);
						fWeight			= nucAlphas[nucPosOffset]*fR0Weight;
						rWeight			= nucAlphas[nucPosOffset]*rR0Weight;
						if (position >= 1) {
							fWeight *= dinucAlphas[fDinucPosOffset];
							rWeight *= dinucAlphas[rDinucPosOffset];
						}
						for (int cgf=0; cgf<totFeatures; cgf++) {
							if (cgf<totNucFeatures) {
								fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
								rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
							}
							if (cgf<totDinucFeatures) {
								fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
								rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
							}
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fWeight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rWeight;
							}
						}
						fwdUpdate = fsOffsets[currOffset][currSubFeat]*fWeight;
						revUpdate = rsOffsets[currOffset][currSubFeat]*rWeight;
						fsWork[currOffset][fNextSubFeat]									+= fwdUpdate;
						rsWork[currOffset][rNextSubFeat]									+= revUpdate;
						fsNucGradientsWork[nucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
						rsNucGradientsWork[nucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
						fsHessianWork[nucPosOffset][nucPosOffset][currOffset][fNextSubFeat] += fwdUpdate;
						rsHessianWork[nucPosOffset][nucPosOffset][currOffset][rNextSubFeat] += revUpdate;
												
						if (position>=1) {
							fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat]				+= fwdUpdate;
							rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat]				+= revUpdate;
							fDinucPosOffset += 4*k;
							rDinucPosOffset += 4*k;
							fsHessianWork[fDinucPosOffset][fDinucPosOffset][currOffset][fNextSubFeat]	+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][rDinucPosOffset][currOffset][rNextSubFeat]	+= revUpdate;
							fsHessianWork[fDinucPosOffset][nucPosOffset][currOffset][fNextSubFeat]		+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][nucPosOffset][currOffset][rNextSubFeat]		+= revUpdate;
							for (int cgf=0; cgf<totNucFeatures; cgf++) {
								fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
								revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
								if (fwdUpdate==0 && revUpdate==0)	continue;
								fsHessianWork[nucPosOffset][cgf][currOffset][fNextSubFeat]				+= fwdUpdate;
								rsHessianWork[nucPosOffset][cgf][currOffset][rNextSubFeat]				+= revUpdate;
								fsHessianWork[fDinucPosOffset][cgf][currOffset][fNextSubFeat]			+= fwdUpdate;
								rsHessianWork[rDinucPosOffset][cgf][currOffset][rNextSubFeat]			+= revUpdate;
							}
							for (int cgf=0; cgf<totDinucFeatures; cgf++) {
								fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
								revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
								if (fwdUpdate==0 && revUpdate==0)	continue;
								tempIdx = cgf+4*k;
								fsHessianWork[tempIdx][nucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
								rsHessianWork[tempIdx][nucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
								fsHessianWork[fDinucPosOffset][tempIdx][currOffset][fNextSubFeat]		+= fwdUpdate;
								rsHessianWork[rDinucPosOffset][tempIdx][currOffset][rNextSubFeat]		+= revUpdate;
							}
						} else {
							for (int cgf=0; cgf<totNucFeatures; cgf++) {
								fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
								revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
								if (fwdUpdate==0 && revUpdate==0)	continue;
								fsHessianWork[nucPosOffset][cgf][currOffset][fNextSubFeat] += fwdUpdate;
								rsHessianWork[nucPosOffset][cgf][currOffset][rNextSubFeat] += revUpdate;
							}
							for (int cgf=0; cgf<totDinucFeatures; cgf++) {
								fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
								revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
								if (fwdUpdate==0 && revUpdate==0)	continue;
								tempIdx = cgf+4*k;
								fsHessianWork[nucPosOffset][tempIdx][currOffset][fNextSubFeat] += fwdUpdate;
								rsHessianWork[nucPosOffset][tempIdx][currOffset][rNextSubFeat] += revUpdate;
							}	
						}
					}
					for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							if (cgf<totNucFeatures) {
								fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							if (cgf<totDinucFeatures) {
								fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
							}
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
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
		
		//Loop over right fixed region
		for (int index=l+flankLength; index<l+2*flankLength; index++) {								//R0, NUC, DINUC
			startOffset 		= (index-k+1 < 0) ? 0 : index-k+1;
			endOffset 			= (index > maxFrames-1) ? maxFrames-1 : index;
			for (int i=0; i<maxFrames; i++) {
				for (int j=0; j<maxSubFeatures; j++) {
					fsWork[i][j] = 0;
					rsWork[i][j] = 0;
				}
			}
			for (int k=0; k<totNucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsNucGradientsWork[k][i][j] = 0;
						rsNucGradientsWork[k][i][j] = 0;
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						rsDinucGradientsWork[k][i][j] = 0;
						fsDinucGradientsWork[k][i][j] = 0;
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
			
			//Find new base
			int fNewBase = (int) fsRFlank & 3;
			int rNewBase = (int) rsRFlank & 3;
			fsRFlank >>= 2;
			rsRFlank >>= 2;
			for (int currSubFeat=0; currSubFeat<maxSubFeatures; currSubFeat++) {
				fNewWorkFeat= (currSubFeat<<2) | fNewBase;
				rNewWorkFeat= (currSubFeat<<2) | rNewBase;
				fNextSubFeat= (int) (fNewWorkFeat & subFeatureMask);
				rNextSubFeat= (int) (rNewWorkFeat & subFeatureMask);
				fR0Weight	= 1;
				rR0Weight	= 1;
				if (index < l+flankLength+r0k-1) {
					fR0Weight = r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight = r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
				}

				for (int currOffset=0; currOffset<startOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
						}
					}
					//Next, update non-gradient sums
					fsWork[currOffset][fNextSubFeat] += fsOffsets[currOffset][currSubFeat]*fR0Weight;
					rsWork[currOffset][rNextSubFeat] += rsOffsets[currOffset][currSubFeat]*rR0Weight;
				}
				for (int currOffset=startOffset; currOffset<=endOffset; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					position		= index-currOffset;
					fNucPosOffset	= position*4 + fNewBase;
					rNucPosOffset	= position*4 + rNewBase;
					fDinucPosOffset	= (position-1)*16 + (int) (fNewWorkFeat & 15);
					rDinucPosOffset	= (position-1)*16 + (int) (rNewWorkFeat & 15);
					fWeight			= nucAlphas[fNucPosOffset]*fR0Weight;
					rWeight			= nucAlphas[rNucPosOffset]*rR0Weight;
					if (position >= 1) {
						fWeight *= dinucAlphas[fDinucPosOffset];
						rWeight *= dinucAlphas[rDinucPosOffset];
					}
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fWeight;
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rWeight;
						}
					}
					fwdUpdate = fsOffsets[currOffset][currSubFeat]*fWeight;
					revUpdate = rsOffsets[currOffset][currSubFeat]*rWeight;
					fsWork[currOffset][fNextSubFeat]									+= fwdUpdate;
					rsWork[currOffset][rNextSubFeat]									+= revUpdate;
					fsNucGradientsWork[fNucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
					rsNucGradientsWork[rNucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
					fsHessianWork[fNucPosOffset][fNucPosOffset][currOffset][fNextSubFeat] += fwdUpdate;
					rsHessianWork[rNucPosOffset][rNucPosOffset][currOffset][rNextSubFeat] += revUpdate;
					
					if (position>=1) {
						fsDinucGradientsWork[fDinucPosOffset][currOffset][fNextSubFeat]				+= fwdUpdate;
						rsDinucGradientsWork[rDinucPosOffset][currOffset][rNextSubFeat]				+= revUpdate;
						fDinucPosOffset += 4*k;
						rDinucPosOffset += 4*k;
						fsHessianWork[fDinucPosOffset][fDinucPosOffset][currOffset][fNextSubFeat]	+= fwdUpdate;
						rsHessianWork[rDinucPosOffset][rDinucPosOffset][currOffset][rNextSubFeat]	+= revUpdate;
						fsHessianWork[fDinucPosOffset][fNucPosOffset][currOffset][fNextSubFeat]		+= fwdUpdate;
						rsHessianWork[rDinucPosOffset][rNucPosOffset][currOffset][rNextSubFeat]		+= revUpdate;
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							fsHessianWork[fNucPosOffset][cgf][currOffset][fNextSubFeat]				+= fwdUpdate;
							rsHessianWork[rNucPosOffset][cgf][currOffset][rNextSubFeat]				+= revUpdate;
							fsHessianWork[fDinucPosOffset][cgf][currOffset][fNextSubFeat]			+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][cgf][currOffset][rNextSubFeat]			+= revUpdate;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							tempIdx = cgf+4*k;
							fsHessianWork[tempIdx][fNucPosOffset][currOffset][fNextSubFeat]			+= fwdUpdate;
							rsHessianWork[tempIdx][rNucPosOffset][currOffset][rNextSubFeat]			+= revUpdate;
							fsHessianWork[fDinucPosOffset][tempIdx][currOffset][fNextSubFeat]		+= fwdUpdate;
							rsHessianWork[rDinucPosOffset][tempIdx][currOffset][rNextSubFeat]		+= revUpdate;
						}
					} else {
						for (int cgf=0; cgf<totNucFeatures; cgf++) {
							fwdUpdate = fsNucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsNucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							fsHessianWork[fNucPosOffset][cgf][currOffset][fNextSubFeat] += fwdUpdate;
							rsHessianWork[rNucPosOffset][cgf][currOffset][rNextSubFeat] += revUpdate;
						}
						for (int cgf=0; cgf<totDinucFeatures; cgf++) {
							fwdUpdate = fsDinucGradients[cgf][currOffset][currSubFeat]*fWeight;
							revUpdate = rsDinucGradients[cgf][currOffset][currSubFeat]*rWeight;
							if (fwdUpdate==0 && revUpdate==0)	continue;
							tempIdx = cgf+4*k;
							fsHessianWork[fNucPosOffset][tempIdx][currOffset][fNextSubFeat] += fwdUpdate;
							rsHessianWork[rNucPosOffset][tempIdx][currOffset][rNextSubFeat] += revUpdate;
						}	
					}
				}
				for (int currOffset=endOffset+1; currOffset<maxFrames; currOffset++) {
					if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
					for (int cgf=0; cgf<totFeatures; cgf++) {
						if (cgf<totNucFeatures) {
							fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						if (cgf<totDinucFeatures) {
							fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
							rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
						}
						for (int cgf2=0; cgf2<=cgf; cgf2++) {
							fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
							rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
						}
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
						fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
						rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
					}
				}
			}
			for (int k=0; k<totDinucFeatures; k++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
						rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
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
		
		if (r0k-1 > flankLength) {
			for (int index=0; index<r0k-1-flankLength; index++) {
				for (int i=0; i<maxFrames; i++) {
					for (int j=0; j<maxSubFeatures; j++) {
						fsWork[i][j] = 0;
						rsWork[i][j] = 0;
					}
				}
				for (int k=0; k<totNucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsNucGradientsWork[k][i][j] = 0;
							rsNucGradientsWork[k][i][j] = 0;
						}
					}
				}
				for (int k=0; k<totDinucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							rsDinucGradientsWork[k][i][j] = 0;
							fsDinucGradientsWork[k][i][j] = 0;
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
					fR0Weight	= r0Alphas[(int) (fNewWorkFeat&r0FeatureMask)];
					rR0Weight	= r0Alphas[(int) reverseComplement((rNewWorkFeat&r0FeatureMask), r0k)];
					for (int currOffset=0; currOffset<maxFrames; currOffset++) {
						if (fsOffsets[currOffset][currSubFeat]==0 && rsOffsets[currOffset][currSubFeat]==0)	continue;
						for (int cgf=0; cgf<totFeatures; cgf++) {
							if (cgf<totNucFeatures) {
								fsNucGradientsWork[cgf][currOffset][fNextSubFeat] += fsNucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsNucGradientsWork[cgf][currOffset][rNextSubFeat] += rsNucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							if (cgf<totDinucFeatures) {
								fsDinucGradientsWork[cgf][currOffset][fNextSubFeat] += fsDinucGradients[cgf][currOffset][currSubFeat]*fR0Weight;
								rsDinucGradientsWork[cgf][currOffset][rNextSubFeat] += rsDinucGradients[cgf][currOffset][currSubFeat]*rR0Weight;
							}
							for (int cgf2=0; cgf2<=cgf; cgf2++) {
								fsHessianWork[cgf][cgf2][currOffset][fNextSubFeat] += fsHessian[cgf][cgf2][currOffset][currSubFeat]*fR0Weight;
								rsHessianWork[cgf][cgf2][currOffset][rNextSubFeat] += rsHessian[cgf][cgf2][currOffset][currSubFeat]*rR0Weight;
							}
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
							fsNucGradients[k][i][j] = fsNucGradientsWork[k][i][j];
							rsNucGradients[k][i][j] = rsNucGradientsWork[k][i][j];
						}
					}
				}
				for (int k=0; k<totDinucFeatures; k++) {
					for (int i=0; i<maxFrames; i++) {
						for (int j=0; j<maxSubFeatures; j++) {
							fsDinucGradients[k][i][j] = fsDinucGradientsWork[k][i][j];
							rsDinucGradients[k][i][j] = rsDinucGradientsWork[k][i][j];
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
		}
				
		Z = 0;
		for (int i=0; i<maxFrames; i++) {
			Z += Array.sum(fsOffsets[i]) + Array.sum(rsOffsets[i]);
		}
		//Sum over frames to get total gradient
		nucGradients	= new double[totNucFeatures];
		dinucGradients	= new double[totDinucFeatures];
		hessian			= (isNSBinding) ? new double[totFeatures+1][totFeatures+1] : new double[totFeatures][totFeatures];
		for (int cgf=0; cgf<totFeatures; cgf++) {
			for (int currOffset=0; currOffset<maxFrames; currOffset++) {				
				if (cgf<totNucFeatures) {
					nucGradients[cgf] += Array.sum(fsNucGradients[cgf][currOffset]) 
							+ Array.sum(rsNucGradients[cgf][currOffset]);
				}
				if (cgf<totDinucFeatures) {
					dinucGradients[cgf] += Array.sum(fsDinucGradients[cgf][currOffset]) 
							+ Array.sum(rsDinucGradients[cgf][currOffset]);	
				}
				for (int cgf2=0; cgf2<=cgf; cgf2++) {
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
		fsNucGradients		= null;
		rsNucGradients		= null;
		fsDinucGradients	= null;
		rsDinucGradients	= null;
		fsNucGradientsWork	= null;
		rsNucGradientsWork	= null;
		fsDinucGradientsWork= null;
		rsDinucGradientsWork= null;
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
		return Array.clone(dinucGradients);
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
