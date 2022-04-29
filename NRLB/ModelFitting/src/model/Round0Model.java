package model;

import base.*;

public class Round0Model extends Model{
	private boolean isFlank;
	private int l, k, maxOffsets, maxFeatures, maxSubFeatures;
	private int nCount;
	private long flankingSequence, leftFlank, rightFlank, mask;
	private double Z, logZ;
	private String lFlank, rFlank, filePath = "NA";
	private double[] betas, alphas, sumkXphi, gradient;
	private long[] probes;
	private int[] counts;
	private Round0Results results = null;
	
	public Round0Model(Data data, int k, boolean isFlank) {
		l				= data.l;
		this.k			= k;
		maxFeatures		= (int) Math.pow(4, k);
		mask			= maxFeatures-1;
		maxSubFeatures	= maxFeatures/4;
		this.isFlank	= isFlank;
		maxOffsets		= (isFlank) ? l+k-1 : l-k+1;
		if (isFlank) {
			lFlank			= data.leftFlank;
			rFlank			= data.rightFlank;
			leftFlank		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
			leftFlank		= leftFlank & (maxSubFeatures-1);
			rightFlank		= reverse((new Sequence(rFlank, 0, rFlank.length())).getValue(), rFlank.length());
			rightFlank		= reverse(rightFlank & (maxSubFeatures-1), k-1);
			flankingSequence= (leftFlank << 2*(l+k-1)) | rightFlank;
			rightFlank		= reverse(rightFlank, k-1);
		}

		nCount			= data.nCount;
		probes			= data.probes;
		counts			= data.counts;
		initDataMatrices();

		setParams(new double[maxFeatures]);
	}
	
	public Round0Model(String resultsFile, int l, int k, boolean isFlank, 
			String lFlank, String rFlank) {
		results			= new Round0Results(resultsFile);
		filePath		= resultsFile;
		this.l			= l;
		if (k<=0) {		//Check if k is undefined
			this.k		= results.getBestFit().k;
		} else {
			this.k		= k;			
		}
		maxFeatures		= (int) Math.pow(4, this.k);
		mask			= maxFeatures-1;
		maxSubFeatures	= maxFeatures/4;
		this.isFlank	= isFlank;
		maxOffsets		= (isFlank) ? l+this.k-1 : l-this.k+1;
		if (isFlank) {
			if (lFlank!=null) {
				if (!lFlank.equals(results.lFlank)) {
					System.err.println("WARNING: THE LEFT FLANKING SEQUENCE PROVIDED DOES NOT MATCH THE SEQUENCE THE ROUND0 MODEL WAS CONSTRUCTED ON");
				}
			} else {
				lFlank = results.lFlank;
			}
			if (rFlank!=null) {
				if (!rFlank.equals(results.rFlank)) {
					System.err.println("WARNING: THE RIGHT FLANKING SEQUENCE PROVIDED DOES NOT MATCH THE SEQUENCE THE ROUND0 MODEL WAS CONSTRUCTED ON");
				}
			} else {
				rFlank = results.rFlank;
			}
			if ((rFlank.length()<this.k-1) || (lFlank.length()<this.k-1)) {
				throw new IllegalArgumentException("Round0Model: Input Left/Right Flanks are too short.");
			}
			
			this.lFlank		= lFlank;
			this.rFlank		= rFlank;
			leftFlank		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
			leftFlank		= leftFlank & (maxSubFeatures-1);
			rightFlank		= reverse((new Sequence(rFlank, 0, rFlank.length())).getValue(), rFlank.length());
			rightFlank		= reverse(rightFlank & (maxSubFeatures-1), this.k-1);
			flankingSequence= (leftFlank << 2*(l+this.k-1)) | rightFlank;
			rightFlank		= reverse(rightFlank, this.k-1);
		}
		setParams(results.getFit(this.k, isFlank).betas);
	}
	
	@Override
	public void replaceData(Object o) {
		Data input;
		if(o instanceof Data) {
			input = (Data) o;
		} else {
			throw new IllegalArgumentException("replaceData must use Multinomial Data as input!");
		}
		if (input.l!=l || !lFlank.equals(input.leftFlank) || !rFlank.equals(input.rightFlank)) {
			throw new IllegalArgumentException("New dataset is not similar to the old one!");
		}
		nCount = input.nCount;
		probes = input.probes;
		counts = input.counts;
		initDataMatrices();
	}

	public double likelihoodNormalizer() {
		return 1.0/nCount;
	}

	public double maxLikelihood() {
		double output	= 0;
		for (int i : counts) {
			output -= i*Math.log(i);
		}
		output += nCount*Math.log(nCount);
		return output;
	}

	@Override
	public void normalForm() {
		double maxVal	= Array.max(betas);
		double[] output = Array.clone(betas);
		
		for (int i=0; i<output.length; i++) {
			output[i] -= maxVal;
		}
		setParams(output);
	}
	
	public String modelPath() {
		return filePath;
	}
	
	public int getNDimensions() {
		return maxFeatures;
	}
	
	public int getTotFeatures() {
		return maxFeatures;
	}

	public void setParams(double[] position) {
		betas	= Array.clone(position);
		alphas	= Array.exp(betas);
		calcZ();
	}

	public double[] getPositionVector() {
		return Array.clone(betas);
	}
	
	public double[] getBetas() {
		return Array.clone(betas);
	}
	
	public double[] getAlphas() {
		return Array.clone(alphas);
	}

	//Does nothing in this model.
	public double[] shiftBetas(double[] originalBetas, int shiftPositions) {
		return originalBetas;
	}

	//No concept of an 'orthogonal step' here.
	public double[] orthogonalStep(double[] currPos, int position, double stepSize) {
		double[] output = Array.clone(currPos);
		output[position] += stepSize;
		return output;
	}

	@Override
	public double functionEval() throws Exception {
		double output = nCount*logZ;
		for (int phi=0; phi<maxFeatures; phi++) {
			output -= betas[phi]*sumkXphi[phi];
		}
		return output;
	}

	@Override
//	public CompactGradientOutput gradientEval() throws Exception {
//		double fVal = functionEval();
//		double[] output = gradientFiniteDifferences(betas, 1E-5);
//		return (new CompactGradientOutput(fVal, output));
//	}
	public CompactGradientOutput gradientEval() throws Exception {
		gradient	= Array.subtract(gradZ(), sumkXphi);
		double fVal	= nCount*logZ;
		for (int phi=0; phi<maxFeatures; phi++) {
			fVal -= betas[phi]*sumkXphi[phi];
		}
		return (new CompactGradientOutput(fVal, gradient));
	}

	@Override
	public CompactGradientOutput getGradient() {
		return (new CompactGradientOutput(0, gradient));
	}

	@Override
	public CompactGradientOutput hessianEval() throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Fit generateFit(double[] seed) {
		return (new Round0Fit(this, seed));
	}

	private void initDataMatrices() {
		int currCount;
		long currSeq;
		sumkXphi 	= new double[maxFeatures];
		
		if (isFlank) {
			for (int i=0; i<probes.length; i++) {
				currSeq		= probes[i];
				currSeq		= flankingSequence | (currSeq << 2*(k-1));
				currCount	= counts[i]; 
				for (int offset = 0; offset < maxOffsets; offset++) {
					sumkXphi[(int) (currSeq & mask)] += currCount;
					currSeq = currSeq >> 2;
				}
			}
		} else {
			for (int i=0; i<probes.length; i++) {
				currSeq		= probes[i];
				currCount	= counts[i]; 
				for (int offset = 0; offset < maxOffsets; offset++) {
					sumkXphi[(int) (currSeq & mask)] += currCount;
					currSeq = currSeq >> 2;
				}
			}
		}
	}
	
	//calculate the partition function
	private void calcZ(){
		double[] runningTotal = new double[maxSubFeatures];
		double[] work;
		long subFeatureMask   = (long) maxSubFeatures-1;
		long newWorkFeat;
		
		if (!isFlank) {
			//Initialize
			for (long currFeat = 0; currFeat< maxFeatures; currFeat++) {
				runningTotal[(int) (currFeat & subFeatureMask)] += alphas[(int) currFeat];
			}
			//Consider remaining offsets
			for (int offset = 1; offset < l-k+1; offset++) {
				work = new double[maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					for (long newBase=0; newBase<4; newBase++) {
						newWorkFeat = (currSubFeat<<2) | newBase;
						work[(int) (newWorkFeat & subFeatureMask)] += runningTotal[(int) currSubFeat]*alphas[(int) newWorkFeat];
					}
				}
				runningTotal = work;
			}
		} else {
			long rFlank = rightFlank;
			
			//Initialize
			runningTotal[(int) (leftFlank & subFeatureMask)] = 1;
			//Consider remaining offsets
			for (int offset = 0; offset < l; offset++) {
				work = new double[maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					for (long newBase=0; newBase<4; newBase++) {
						newWorkFeat = (currSubFeat<<2) | newBase;
						work[(int) (newWorkFeat & subFeatureMask)] += runningTotal[(int) currSubFeat]*alphas[(int) newWorkFeat];
					}
				}
				runningTotal = work;
			}
			
			//Now consider fixed right flank bases
			for (int offset = 0; offset < k-1; offset++) {
				work = new double[maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					newWorkFeat = (currSubFeat<<2) | (rFlank & 3);
					work[(int) (newWorkFeat & subFeatureMask)] += runningTotal[(int) currSubFeat]*alphas[(int) newWorkFeat];
				}
				runningTotal = work;
				rFlank >>= 2;
			}
		}
		Z = Array.sum(runningTotal);
		logZ = Math.log(Z);
	}
	
	private double[] gradZ(){
		double newAlpha;
		double[] runningTotal	= new double[maxSubFeatures];
		double[] work, output;
		double[][] gradTotal	= new double[maxFeatures][maxSubFeatures];
		double[][] gradWork;
		long subFeatureMask		= (long) maxSubFeatures-1;
		long newWorkFeat, newSubFeat;
		
		if (!isFlank) {
			//Initialize
			for (long currFeat = 0; currFeat< maxFeatures; currFeat++) {
				newSubFeat = currFeat & subFeatureMask;
				runningTotal[(int) newSubFeat] += alphas[(int) currFeat];
				gradTotal[(int) currFeat][(int) newSubFeat] += alphas[(int) currFeat];
			}
			//Consider remaining offsets
			for (int offset = 1; offset < l-k+1; offset++) {
				work	= new double[maxSubFeatures];
				gradWork= new double[maxFeatures][maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					if (runningTotal[(int) currSubFeat]==0)		continue;
					for (long newBase=0; newBase<4; newBase++) {
						newWorkFeat = (currSubFeat<<2) | newBase;
						newSubFeat	= newWorkFeat & subFeatureMask;
						newAlpha	= alphas[(int) newWorkFeat];
						//Update all gradient features
						for (int cgf=0; cgf<maxFeatures; cgf++) {
							gradWork[cgf][(int) newSubFeat] += gradTotal[cgf][(int) currSubFeat]*newAlpha;
						}
						//Update non-gradient features
						work[(int) (newWorkFeat & subFeatureMask)] += runningTotal[(int) currSubFeat]*newAlpha;
						//Add new gradient feature
						gradWork[(int) newWorkFeat][(int) newSubFeat] += runningTotal[(int) currSubFeat]*newAlpha;
					}
				}
				runningTotal= work;
				gradTotal	= gradWork;
			}
		} else {
			long rFlank = rightFlank;
			
			//Initialize
			runningTotal[(int) (leftFlank & subFeatureMask)] = 1;
			//Consider remaining offsets
			for (int offset = 0; offset < l; offset++) {
				work	= new double[maxSubFeatures];
				gradWork= new double[maxFeatures][maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					if (runningTotal[(int) currSubFeat]==0)		continue;
					for (long newBase=0; newBase<4; newBase++) {
						newWorkFeat = (currSubFeat<<2) | newBase;
						newSubFeat	= newWorkFeat & subFeatureMask;
						newAlpha	= alphas[(int) newWorkFeat];
						//Update all gradient Features
						for (int cgf=0; cgf<maxFeatures; cgf++) {
							gradWork[cgf][(int) newSubFeat] += gradTotal[cgf][(int) currSubFeat]*newAlpha;
						}
						//Update non-gradient features
						work[(int) newSubFeat] += runningTotal[(int) currSubFeat]*newAlpha;
						//Add new gradient feature
						gradWork[(int) newWorkFeat][(int) newSubFeat] += runningTotal[(int) currSubFeat]*newAlpha;
					}
				}
				runningTotal= work;
				gradTotal	= gradWork;
			}
			
			//Now consider fixed right flank bases
			for (int offset = 0; offset < k-1; offset++) {
				work	= new double[maxSubFeatures];
				gradWork= new double[maxFeatures][maxSubFeatures];
				for (long currSubFeat = 0; currSubFeat<maxSubFeatures; currSubFeat++) {
					if (runningTotal[(int) currSubFeat]==0)		continue;
					newWorkFeat = (currSubFeat<<2) | (rFlank & 3);
					newSubFeat	= newWorkFeat & subFeatureMask;
					newAlpha	= alphas[(int) newWorkFeat];
					//Update all gradient Features
					for (int cgf=0; cgf<maxFeatures; cgf++) {
						gradWork[cgf][(int) newSubFeat] += gradTotal[cgf][(int) currSubFeat]*newAlpha;
					}
					work[(int) newSubFeat] += runningTotal[(int) currSubFeat]*newAlpha;
					//Add new gradient feature
					gradWork[(int) newWorkFeat][(int) newSubFeat] += runningTotal[(int) currSubFeat]*newAlpha;
				}
				runningTotal= work;
				gradTotal	= gradWork;
				rFlank >>= 2;
			}
		}
		Z		= Array.sum(runningTotal);
		logZ	= Math.log(Z);
		output	= new double[maxFeatures];
		for (int i=0; i<maxFeatures; i++) {
			output[i] = nCount*Array.sum(gradTotal[i])/Z;
		}
		return output;
	}
	
	//Evaluate pi0 for a given probe
	public double getSeqProb(long input) {
		double output = -logZ;		//subtract Z
		if (isFlank) {				//If considering flanks, properly offset the input sequence
			input = flankingSequence | (input << 2*(k-1));
		}
		
		for (int offset=0; offset<maxOffsets; offset++) {
			output += betas[(int) (input & mask)];
			input >>= 2;
		}	
		return Math.exp(output);
	}
	
	public double getZ() {
		calcZ();
		return Z;
	}
	
	public boolean isFlank() {
		return isFlank;
	}
	
	public int getK() {
		return k;
	}
	
	public int getL() {
		return l;
	}
	
	public int getNCount() {
		return nCount;
	}	
}
