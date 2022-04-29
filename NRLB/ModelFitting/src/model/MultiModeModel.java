package model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import base.*;

public class MultiModeModel extends Model{
	private boolean isDinuc, isShape, isNSBinding, isMemSafe, modeRegression;
	private int l, nCount, nThreads, totFeatures, flankLength;
	private int nShapeClasses, nModes, nonNSFeatures, type, totalFrames;
	private String lFlank, rFlank;
	private double Z_FFM;					//Z of the full model, with NS Binding
	private double nsBindingValue;			//THIS IS THE BETA_NS REPRESENTATION
	private double lambda = 0;				//L2 Regularization Parameter
	private boolean[] isActive;
	private int[] R1Counts, reverseMap, ks;
	private long[] R1Probes;
	public double[] betas, modeAffinities;
	private double[] gradients, R1R0Prob, activeFeatures;
	private String[] nucSymmetries, dinucSymmetries;
	private int[][] swThreadRange, dpThreadRange, kappaIdx;
	private int[][][] betaIdx;
	private ArrayList<SingleModeSW> modes;
	private Data data;
	private Shape shapeModel;
	private Round0Model R0Model;
	private ExecutorService swPool, dpPool;
	private Matrix[] featureM;

	public MultiModeModel(int nThreads, Shape shapeModel, Data data, boolean isFlank, 
			int flankLength, boolean isDinuc, boolean isShape, boolean isNSBinding,
			int[] ks, String[] nucSymmetries, String[] dinucSymmetries) {
		int offset = 0;
		SingleModeSW tempMode;
		int[] map1, map2;
		double[] work;
		ArrayList<double[]> matrix = new ArrayList<double[]>();
		
		if (!isFlank || flankLength==0) {
			flankLength = 0;
			isFlank = false;
		}
		
		//Get Round0 Information
		R0Model	 			= data.R0Model;
		//Load Shape Table Information
		if (shapeModel!=null) {
			this.shapeModel	= shapeModel;
			nShapeClasses	= shapeModel.nShapeFeatures();	
		}
		//Load Data
		this.data			= data;
		l					= data.l;
		nCount				= data.nCount;
		R1Counts			= data.counts;
		R1Probes			= data.probes;
		R1R0Prob			= data.R0Prob;
		//Load Parameters
		this.flankLength	= flankLength;
		lFlank				= data.leftFlank;
		rFlank				= data.rightFlank;
		this.nThreads		= nThreads;
		this.isDinuc		= isDinuc;
		this.isShape		= isShape;
		this.isNSBinding	= isNSBinding;
		
		//Create multiple modes - first check to see if mode lengths match
		this.ks				= ks;
		nModes				= ks.length;
		modeAffinities		= (isNSBinding) ? new double[nModes+1] : new double[nModes];
		if (nucSymmetries!=null && nucSymmetries.length!=nModes) {
			throw new IllegalArgumentException("Nucleotide Symmetry array length does not match the number of modes!");
		}
		this.nucSymmetries	= nucSymmetries;
		if (dinucSymmetries!=null && dinucSymmetries.length!=nModes) {
			throw new IllegalArgumentException("Dinucleotide Symmetry array length does not match the number of modes!");
		}
		this.dinucSymmetries= dinucSymmetries;
		//Create Modes, find total number of features and build maps
		nonNSFeatures		= 0;
		totalFrames			= 0;
		modes 				= new ArrayList<SingleModeSW>(nModes);
		reverseMap			= null;
		betaIdx				= new int[nModes][4][2];
		kappaIdx			= new int[nModes][2];
		isActive			= new boolean[nModes];
		for (int i=0; i<nModes; i++) {
			kappaIdx[i][0]	= totalFrames;
			tempMode		= new SingleModeSW(shapeModel, R0Model, l, ks[i], 
					isFlank, flankLength, lFlank, rFlank, isDinuc, isShape);
			tempMode.setOffsets(offset, totalFrames);
			totalFrames		+= 2*tempMode.maxFrames;
			kappaIdx[i][1]	= totalFrames-1;
			nonNSFeatures	+= tempMode.nFeatures;
			modes.add(tempMode);
			
			//Build inversion map and beta index map
			map1			= new int[4*ks[i]];		
			for (int j=0; j<4*ks[i]; j++) {
				map1[j]		= offset + j;
			}
			betaIdx[i][0][0]= offset;
			betaIdx[i][1][0]= offset;
			offset			+= 4*ks[i];
			betaIdx[i][1][1]= offset - 1;
			map1			= Array.blockReverse(map1, 4);
			if (isDinuc) {
				map2		= new int[16*(ks[i]-1)];
				for (int j=0; j<16*(ks[i]-1); j++) {
					map2[j] = offset + j;
				}
				betaIdx[i][2][0]= offset; 
				offset			+= 16*(ks[i]-1);
				betaIdx[i][2][1]= offset-1;
				map2			= Array.blockReverse(map2, 16);
				map1			= Array.cat(map1, map2);
			}
			if (isShape) {		//do the same for shape features
				map2			= new int[nShapeClasses*ks[i]];
				for (int j=0; j<nShapeClasses*ks[i]; j++) {
					map2[j] 	= offset + j;
				}
				betaIdx[i][3][0]= offset;
				offset			+= nShapeClasses*ks[i];
				betaIdx[i][3][1]= offset - 1;
				map2			= Array.blockReverse(map2, nShapeClasses);
				map1			= Array.cat(map1, map2);			
			}
			betaIdx[i][0][1]	= offset-1;
			reverseMap			= Array.cat(reverseMap, map1);
			isActive[i]			= true;
		}
		//Find the total number of features
		totFeatures = (isNSBinding) ? nonNSFeatures + 1 : nonNSFeatures;
		if (isNSBinding) {
			reverseMap = Array.cat(reverseMap, new int[]{reverseMap.length});
		}
		//All features are active initially
		activeFeatures = new double[totFeatures];
		for (int i=0; i<totFeatures; i++) {
			activeFeatures[i] = 1;
		}
		
		offset = 0;
		//Build symmetry matrix for overall model
		for (int i=0; i<nModes; i++) {
			if (nucSymmetries==null || nucSymmetries[i].equals("null")) {
				symmetryHelper(null, 4, 1, matrix, 4*ks[i], offset);
			} else {
				symmetryHelper(nucSymmetries[i], 4, 1, matrix, 4*ks[i], offset);
			}
			offset += 4*ks[i];
			if (isDinuc) {
				if (dinucSymmetries==null || dinucSymmetries[i].equals("null")) {
					symmetryHelper(null, 16, 2, matrix, 16*(ks[i]-1), offset);
				} else {
					symmetryHelper(dinucSymmetries[i], 16, 2, matrix, 16*(ks[i]-1), offset);
				}
				offset += 16*(ks[i]-1);
			}
			if (isShape) {
				if (nucSymmetries==null || nucSymmetries[i].equals("null")) {
					symmetryHelper(null, nShapeClasses, 0, matrix, nShapeClasses*ks[i], offset);
				} else {
					symmetryHelper(nucSymmetries[i], nShapeClasses, 0, matrix, nShapeClasses*ks[i], offset);
				}
				offset += nShapeClasses*ks[i];
			}
		}
		
		if (isNSBinding) {
			work = new double[totFeatures];
			work[totFeatures-1] = 1;
			matrix.add(work);
		}
		featureM = matrixBuilder(matrix, totFeatures);
		M = featureM;
		
		//What type of sliding window should be used?
		if (isFlank) {
			if (isDinuc) {
				if (isShape) {
					type = 3;		//Nuc+Dinuc+Shape
				} else {
					type = 1;		//Nuc+Dinuc
				}
			} else {
				if (isShape) {
					type = 2;		//Nuc+Shape
				} else {
					type = 0;		//Nuc
				}
			}			
		} else {
			if (isDinuc) {
				if (isShape) {
					type = 7;		//Nuc+Dinuc+Shape
				} else {
					type = 5;		//Nuc+Dinuc
				}
			} else {
				if (isShape) {
					type = 6;		//Nuc+Shape
				} else {
					type = 4;		//Nuc
				}
			}
		}

		//Create and Schedule Threads
		if (nThreads!=0) {
			swPool = Executors.newFixedThreadPool(nThreads);
			dpPool = Executors.newFixedThreadPool(nThreads);
			threadSchedule(nThreads);
		}
		
		setParams(new double[totFeatures]);
		
		//Ensure gradient evaluation for all modes is 'memory safe' (less than 5GB)
		isMemSafe = true;
		for (int i=0; i<nModes; i++) {
			if (modes.get(i).nFeatures*modes.get(i).nFeatures*modes.get(i).
					maxFrames*Math.pow(4,  R0Model.getK()-1)*8*4>5E9) {
				isMemSafe = false;
				break;
			}
		}
		
		//Mode regression is inactive inititally
		modeRegression = false;
	}
	
	@Override
	public MultiModeModel clone() {
		boolean isFlank			= (flankLength>0);
		MultiModeModel output	= new MultiModeModel(nThreads, shapeModel, 
				data, isFlank, flankLength, isDinuc, isShape, 
				isNSBinding, ks, nucSymmetries, dinucSymmetries);
		output.setParams(betas);
		output.setLambda(lambda);
		return output;
	}
	
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
		nCount		= input.nCount;
		R1Counts	= input.counts;
		R1Probes	= input.probes;
		R1R0Prob	= input.R0Prob;
		if (nThreads!=0) {
			threadSchedule(nThreads);
		}
	}
	
	public double[] shiftBetas(double[] originalBetas, int shiftPositions) {
		double[] output = null;
		
		if (modeRegression) {
			return originalBetas;
		} else {
			for (int i=0; i<nModes; i++) {
				if (isActive[i]) {
					if (shiftPositions<0) {				//shift left
						output = Array.cat(output, Array.cycleLeft(
								Arrays.copyOfRange(originalBetas, betaIdx[i][1][0], betaIdx[i][1][1]+1), -shiftPositions*4));
						if (isDinuc) {
							output = Array.cat(output, Array.cycleLeft(
									Arrays.copyOfRange(originalBetas, betaIdx[i][2][0], betaIdx[i][2][1]+1), -shiftPositions*16));
						}
						if (isShape) {
							output = Array.cat(output, Array.cycleLeft(
									Arrays.copyOfRange(originalBetas, betaIdx[i][3][0], betaIdx[i][3][1]+1), -shiftPositions*nShapeClasses));
						}
					} else {
						output = Array.cat(output, Array.cycleRight(
								Arrays.copyOfRange(originalBetas, betaIdx[i][1][0], betaIdx[i][1][1]+1), shiftPositions*4));
						if (isDinuc) {
							output = Array.cat(output, Array.cycleRight(
									Arrays.copyOfRange(originalBetas, betaIdx[i][2][0], betaIdx[i][2][1]+1), shiftPositions*16));
						}
						if (isShape) {
							output = Array.cat(output, Array.cycleRight(
									Arrays.copyOfRange(originalBetas, betaIdx[i][3][0], betaIdx[i][3][1]+1), shiftPositions*nShapeClasses));
						}
					}
				} else {
					output = Array.cat(output, Arrays.copyOfRange(originalBetas, betaIdx[i][0][0], betaIdx[i][0][1]+1));
				}
			}
			if (isNSBinding) {
				output = Array.cat(output, new double[]{originalBetas[originalBetas.length-1]});
			}
			return output;
		}
	}
	
	//Not defined here
	public double[] orthogonalStep(double[] currPos, int position, double stepSize) {
		return currPos;
	}
	
	public void setActiveModes(boolean[] isActive) {		
		for (int i=0; i<nModes; i++) {
			this.isActive[i] = isActive[i];
			if (!isActive[i]) {
				for (int j=betaIdx[i][0][0]; j<=betaIdx[i][0][1]; j++) {
					activeFeatures[j] = 0;
				}
			} else {
				for (int j=betaIdx[i][0][0]; j<=betaIdx[i][0][1]; j++) {
					activeFeatures[j] = 1;
				}
			}
		}
	}
	
	public void setActiveDinucs(boolean[] isActive) {
		for (int i=0; i<nModes; i++) {
			if (!isActive[i] && isDinuc) {
				for (int j=betaIdx[i][2][0]; j<=betaIdx[i][2][1]; j++) {
					activeFeatures[j] = 0;
				}
			} else {
				for (int j=betaIdx[i][2][0]; j<=betaIdx[i][2][1]; j++) {
					activeFeatures[j] = 1;
				}				
			}
		}
	}
	
	public void setModeRegression(boolean modeRegression) {
		this.modeRegression = modeRegression;
		if (modeRegression) {
			modeAffinities	= (isNSBinding) ? new double[nModes+1] : new double[nModes];
			M				= null;
		} else {
			M				= featureM;
		}
	}
	
	//setParams does NOT SYMMETRIZE input
	public void setParams(double[] input) {
		double[] output = null;
		
		//If mode regression, need to adjust first set of 4 betas taking symmetry into account
		if (modeRegression) {
			modeAffinities = Array.clone(input);
			M = featureM;
			
			output = symmetrizedModeAffinity(betas, modeAffinities);
			for (int i=0; i<nModes; i++) {
				modes.get(i).setParams(Arrays.copyOfRange(output, betaIdx[i][0][0], 
						betaIdx[i][0][1]+1));
			}
			if (isNSBinding) {
				nsBindingValue = Math.exp(input[input.length-1]);
			}
			
			M = null;
		} else {
			//Copy only the active features from the input vector
			for (int i=0; i<totFeatures; i++) {
				if (activeFeatures[i]==0) {
					input[i] = betas[i];
				}
			}
			//Set feature values for all modes
			for (int i=0; i<nModes; i++) {
				modes.get(i).setParams(Arrays.copyOfRange(input, betaIdx[i][0][0], 
						betaIdx[i][0][1]+1));
			}
			if (isNSBinding) {
				nsBindingValue = Math.exp(input[input.length-1]);
			}
			betas = input;
		}
	}
	
	public double[] symmetrizedModeAffinity(double[] input, double[] modeAffinities) {
		double countOffset;
		double[] temp, output, modeOffset, symSeed = null;
		
		//Build symmetry seed to understand how the symmetry structure works
		for (int i=0; i<nModes; i++) {
			temp = new double[betaIdx[i][0][1]-betaIdx[i][0][0]+1];
			for (int j=0; j<4; j++) {
				temp[j] = 1;
			}
			symSeed = Array.cat(symSeed, temp);
		}
		if (isNSBinding)	symSeed = Array.cat(symSeed, 0);
		symSeed = this.symmetrize(symSeed);
		
		//Now modify input so its symmetrized
		modeOffset = null;
		for (int i=0; i<nModes; i++) {
			temp		= Arrays.copyOfRange(symSeed, betaIdx[i][0][0], 
							betaIdx[i][0][1]+1);
			countOffset	= Array.sum(temp)/4;
			modeOffset	= Array.cat(modeOffset, Array.scalarMultiply(temp, modeAffinities[i]/countOffset));
		}
		if (isNSBinding)	modeOffset = Array.cat(modeOffset, 0);
		
		//Finally modify input array
		output = Array.add(input, modeOffset);
		if (isNSBinding)	output[output.length-1] = modeAffinities[nModes];
		
		return output;
	}
	
	public double getZ() {
		double totalZ = 0;
		
		for (int i=0; i<nModes; i++) {
			totalZ += modes.get(i).getZ();
		}
		totalZ /= R0Model.getZ();
		return (totalZ + nsBindingValue);
	}
	
	public double functionEval() throws InterruptedException, ExecutionException {
		double functionValue;
		List<Future<Double>> threadOutput;
		Set<Callable<Double>> tasks = new HashSet<Callable<Double>>(nThreads);
		
		Z_FFM			= 0;
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			if (dpThreadRange[i][1]!=0) {
				tasks.add(new ThreadedDPEvaluator(0, dpThreadRange[i][0], dpThreadRange[i][1]));
			}
		}
		//Launch Threads
		threadOutput = dpPool.invokeAll(tasks);
		//Sum up value and return
		for (Future<Double> currentThread : threadOutput) {
			Z_FFM += currentThread.get();
		}
		Z_FFM /= R0Model.getZ();
		Z_FFM += nsBindingValue;
		functionValue	= nCount*Math.log(Z_FFM);
		
		reverseBetas();			//Need the proper orientation for sliding windows
		//Assign Threads
		tasks = new HashSet<Callable<Double>>(nThreads);
		for (int i=0; i<nThreads; i++) {
			tasks.add(new ThreadedFunctionEvaluator(swThreadRange[i][0], swThreadRange[i][1]));
		}
		//Launch Threads
		threadOutput = swPool.invokeAll(tasks);
		//Sum up value and return
		for (Future<Double> currentThread : threadOutput) {
			functionValue -= currentThread.get();
		}
		reverseBetas();			//Return to the original state
		if (modeRegression) {
			return functionValue;			
		} else {
			return functionValue + lambda*Math.pow(Array.norm(getPositionVector()),2);
		}
	}
	
	public CompactGradientOutput gradientEval() throws InterruptedException, ExecutionException {
		if (modeRegression) {
			return modeRegressionGradientEval();
		}
		double functionValue;
		List<Future<CompactGradientOutput>> swThreadOutput;
		Set<Callable<Double>> dpTasks = new HashSet<Callable<Double>>(nThreads);
		Set<Callable<CompactGradientOutput>> swTasks = new HashSet<Callable<CompactGradientOutput>>(nThreads);
		CompactGradientOutput threadResult;
		
		//Compute Z_FFM and gradient Z
		gradients			= null;
		Z_FFM				= 0;
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			if (dpThreadRange[i][1]!=0) {
				dpTasks.add(new ThreadedDPEvaluator(1, dpThreadRange[i][0], dpThreadRange[i][1]));
			}
		}
		//Launch Threads
		dpPool.invokeAll(dpTasks);
		//Collect info from thread results
		for (int i=0; i<nModes; i++) {
			Z_FFM			+= modes.get(i).Z.getZ();
			gradients		= Array.cat(gradients, modes.get(i).Z.getNucGradients());
			if (isDinuc)	gradients = Array.cat(gradients, modes.get(i).Z.getDinucGradients());
			if (isShape)	gradients = Array.cat(gradients, modes.get(i).Z.getShapeGradients());
		}
		Z_FFM				= Z_FFM/R0Model.getZ() + nsBindingValue;
		functionValue		= nCount*Math.log(Z_FFM);
		gradients			= Array.scalarMultiply(gradients, nCount/(Z_FFM*R0Model.getZ()));
		if (isNSBinding)	gradients = Array.cat(gradients, new double[]{0});
		//Properly orient betas and gradients for sliding windows 
		reverseBetas();
		reverseGradients();		
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			swTasks.add(new ThreadedGradientEvaluator(swThreadRange[i][0], swThreadRange[i][1]));
		}
		//Launch Threads
		swThreadOutput = swPool.invokeAll(swTasks);
		//Sum up value and return
		for (Future<CompactGradientOutput> currentThread : swThreadOutput) {
			threadResult		= currentThread.get();
			functionValue		-= threadResult.functionValue;
			gradients			= Array.subtract(gradients, threadResult.gradientVector);
		}
		reverseBetas();			//Return to the original state
		reverseGradients();
		//Adjust NSBinding gradient
		if (isNSBinding)	gradients[totFeatures-1] *= nsBindingValue;
		//Adjust for L2 penalty
		functionValue	+= lambda*Math.pow(Array.norm(getPositionVector()),2);
		gradients		= Array.addScalarMultiply(gradients, 2*lambda, betas);
		
		//Scrub inactive regions
		for (int i=0; i<totFeatures; i++) {
			if (activeFeatures[i]==0) {
				gradients[i] = 0;
			}
		}		
		swThreadOutput	= null;
		dpTasks			= null;
		swTasks			= null;
		threadResult	= null;
		
		return (new CompactGradientOutput(functionValue, gradients));
	}
	
	public CompactGradientOutput modeRegressionGradientEval() throws InterruptedException, ExecutionException {
		double functionValue;
		List<Future<CompactGradientOutput>> swThreadOutput;
		Set<Callable<Double>> dpTasks = new HashSet<Callable<Double>>(nThreads);
		Set<Callable<CompactGradientOutput>> swTasks = new HashSet<Callable<CompactGradientOutput>>(nThreads);
		CompactGradientOutput threadResult;
		
		//Compute Z_FFM and gradient Z
		gradients			= new double[nModes];
		Z_FFM				= 0;
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			if (dpThreadRange[i][1]!=0) {
				dpTasks.add(new ThreadedDPEvaluator(1, dpThreadRange[i][0], dpThreadRange[i][1]));
			}
		}
		//Launch Threads
		dpPool.invokeAll(dpTasks);
		//Collect info from thread results
		for (int i=0; i<nModes; i++) {
			Z_FFM			+= modes.get(i).Z.getZ();
			gradients[i]	= modes.get(i).Z.getZ();
		}
		Z_FFM				= Z_FFM/R0Model.getZ() + nsBindingValue;
		functionValue		= nCount*Math.log(Z_FFM);
		gradients			= Array.scalarMultiply(gradients, nCount/(Z_FFM*R0Model.getZ()));
		if (isNSBinding)	gradients = Array.cat(gradients, new double[]{0});
		//Properly orient betas and gradients for sliding windows 
		reverseBetas();
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			swTasks.add(new ThreadedModeGradientEvaluator(swThreadRange[i][0], swThreadRange[i][1]));
		}
		//Launch Threads
		swThreadOutput = swPool.invokeAll(swTasks);
		//Sum up value and return
		for (Future<CompactGradientOutput> currentThread : swThreadOutput) {
			threadResult		= currentThread.get();
			functionValue		-= threadResult.functionValue;
			gradients			= Array.subtract(gradients, threadResult.gradientVector);
		}
		reverseBetas();			//Return to the original state
		//Adjust NSBinding gradient
		if (isNSBinding)	gradients[nModes] *= nsBindingValue;
		swThreadOutput	= null;
		dpTasks			= null;
		swTasks			= null;
		threadResult	= null;
		
		return (new CompactGradientOutput(functionValue, gradients));
	}
	
	@Override
	public CompactGradientOutput getGradient() {
		return (new CompactGradientOutput(0, gradients));
	}
	
	public CompactGradientOutput hessianEval() throws InterruptedException, ExecutionException {
		if (modeRegression) {
			hessian	= Array.clone(this.hessianFiniteDifferences(this.getPositionVector(), 1E-5));
			return (new CompactGradientOutput(0, null, hessian));
		}
		//Ensure 'memory safe' operation
		if (isMemSafe) {
			double functionValue;
			List<Future<CompactGradientOutput>> swThreadOutput;
			Set<Callable<Double>> dpTasks = new HashSet<Callable<Double>>(nThreads);
			Set<Callable<CompactGradientOutput>> swTasks = new HashSet<Callable<CompactGradientOutput>>(nThreads);
			CompactGradientOutput threadResult;
			
			//Compute Z_FFM, gradient Z and hessianZ
			Z_FFM			= 0;
			gradients		= null;
			hessian			= new double[totFeatures][totFeatures];
			//Assign Threads
			for (int i=0; i<nThreads; i++) {
				if (dpThreadRange[i][1]!=0) {
					dpTasks.add(new ThreadedDPEvaluator(2, dpThreadRange[i][0], dpThreadRange[i][1]));
				}
			}
			//Launch Threads
			dpPool.invokeAll(dpTasks);
			
			for (int i=0; i<nModes; i++) {
				Z_FFM			+= modes.get(i).Z.getZ();
				gradients		= Array.cat(gradients, modes.get(i).Z.getNucGradients());
				if (isDinuc)	gradients = Array.cat(gradients, modes.get(i).Z.getDinucGradients());
				if (isShape)	gradients = Array.cat(gradients, modes.get(i).Z.getShapeGradients());
				blockDiagMatrix(hessian, modes.get(i).Z.getHessian(), betaIdx[i][0][0]);
			}
			Z_FFM				= Z_FFM/R0Model.getZ() + nsBindingValue;
			functionValue		= nCount*Math.log(Z_FFM);
			gradients			= Array.scalarMultiply(gradients, 1/R0Model.getZ());
			if (isNSBinding)	gradients = Array.cat(gradients, new double[]{0});
			hessian				= Array.scalarMultiply(hessian, nCount/(Z_FFM*R0Model.getZ()));
			
			//subtract n/Z^2 dZ/dBeta1*dZ/dBeta2
			for (int i=0; i<nonNSFeatures; i++) {
				for (int j=0; j<nonNSFeatures; j++) {
					hessian[i][j] -= nCount/(Z_FFM*Z_FFM)*gradients[i]*gradients[j];
				}
				//nsBinding offset
				if (isNSBinding) {
					hessian[i][nonNSFeatures] -= nCount/(Z_FFM*Z_FFM)*gradients[i];
					hessian[nonNSFeatures][i] -= nCount/(Z_FFM*Z_FFM)*gradients[i];				
				}
			}
			//create proper offset for gradient evaluation
			gradients			= Array.scalarMultiply(gradients, nCount/Z_FFM);
					
			//Properly orient betas and gradients for sliding windows 
			reverseBetas();
			reverseGradients();
			reverseHessian();
			//Assign Threads		
			for (int i=0; i<nThreads; i++) {
				swTasks.add(new ThreadedHessianEvaluator(swThreadRange[i][0], swThreadRange[i][1]));
			}
			//Launch Threads
			swThreadOutput = swPool.invokeAll(swTasks);
			//Sum up value and return
			for (Future<CompactGradientOutput> currentThread : swThreadOutput) {
				threadResult		= currentThread.get();
				functionValue		-= threadResult.functionValue;
				gradients			= Array.subtract(gradients, threadResult.gradientVector);
				for (int i=0; i<nonNSFeatures; i++) {
					for (int j=0; j<nonNSFeatures; j++) {
						hessian[i][j] -= threadResult.hessian[i][j];
					}
					if (isNSBinding) {
						hessian[i][nonNSFeatures] -= threadResult.hessian[i][nonNSFeatures];
						hessian[nonNSFeatures][i] -= threadResult.hessian[nonNSFeatures][i];					
					}
				}
				if (isNSBinding) {
					hessian[nonNSFeatures][nonNSFeatures] += threadResult.hessian[nonNSFeatures][nonNSFeatures];
				}
			}		
			reverseBetas();			//Return to the original state
			reverseGradients();
			reverseHessian();
			//Adjust NSBinding values
			if (isNSBinding) {
				gradients[totFeatures-1] *= nsBindingValue;
				for (int i=0; i<nonNSFeatures; i++) {
					hessian[i][nonNSFeatures] *= nsBindingValue;
					hessian[nonNSFeatures][i] *= nsBindingValue;
				}
				hessian[nonNSFeatures][nonNSFeatures] *= nsBindingValue;
			}
			//Adjust for L2 penalty
			functionValue	+= lambda*Math.pow(Array.norm(getPositionVector()),2);
			gradients		= Array.addScalarMultiply(gradients, 2*lambda, betas);
			for (int i=0; i<totFeatures; i++) {
				hessian[i][i] += 2*lambda;
			}
			
			//Scrub inactive regions
			for (int i=0; i<totFeatures; i++) {
				if (activeFeatures[i]==0) {
					gradients[i] = 0;
					for (int j=0; j<totFeatures; j++) {
						hessian[i][j] = 0;
						hessian[j][i] = 0;
					}
				}
			}
			swThreadOutput	= null;
			dpTasks			= null;
			swTasks			= null;
			threadResult	= null;
			
			return (new CompactGradientOutput(functionValue, gradients, hessian));
		} else {
			System.out.println("Evaluating hessian via finite difference method.");
			hessian	= Array.clone(this.hessianFiniteDifferences(this.getPositionVector(), 1E-5));
			return (new CompactGradientOutput(0, null, hessian));
		}
	}
	
	//Specific implementation of errorEval to adjust eigenvalues 
	@Override
	public double[] errorEval() {
		boolean isNaN				= true;
		int nDims					= this.getNDimensions();
		double tol					= (isMemSafe) ? evCutoff : 5E-9;
		double currError;
		errorBars 					= new double[totFeatures];
		double[] absEV, realEV;
		double[][] hessian, inv, DMatrix;
		Matrix V, D;
		EigenvalueDecomposition ev;
		CompactGradientOutput output= null;
		
		//compute hessian
		try {
			output = hessianEval();
		} catch (Exception e) {
			e.printStackTrace();
		}
		//handle the case where the hessian is undefined
		if (output==null || output.hessian==null) {
			return null;
		}
				
		//Moore-Penrose Psuedoinverse
		//Compute SVD on REDUCED hessian AFTER removing null vectors
		hessian = Array.symmetrize(output.hessian);
		ev		= new EigenvalueDecomposition(new Matrix(hessian));
		//Get singular values
		realEV	= ev.getRealEigenvalues();
		absEV	= new double[totFeatures];
		for (int j=0; j<totFeatures; j++) {
			absEV[j] = Math.abs(realEV[j]- 2*lambda);
		}
		absEV	= Array.maxNormalize(absEV);
		//Recompute and reduce hessian after removing null vectors
		DMatrix = ev.getD().getArray();
		V		= ev.getV().copy();
		for (int l=0; l<totFeatures; l++) {
			if (absEV[l]<tol)	DMatrix[l][l] = 0;
		}
		D 		= new Matrix(DMatrix);
		hessian = (V.times(D.times(V.transpose()))).getArray();
		//Transform hessian into reduced space and symmetrize to be safe
		hessian = Array.symmetrize(compressHessian(hessian));
		
		//Invert reduced hessian
		ev		= new EigenvalueDecomposition(new Matrix(hessian));
		//Get singular values
		realEV	= ev.getRealEigenvalues();
		absEV	= Array.maxNormalize(Array.abs(realEV));
		//Loop over tolerance until matrix is successfully inverted
		while (isNaN) {
			V		= ev.getV().copy();
			D		= ev.getD().copy();
			DMatrix = new double[nDims][nDims];
			//Remove singular eigenvectors and eigenvalues
			for (int j=0; j<nDims; j++) {
				if (absEV[j]<tol) {		
					DMatrix[j][j] = 0;
				} else {
					DMatrix[j][j] = 1/D.get(j,j);
				}
			}
			D = new Matrix(DMatrix);
			//invert hessian
			inv = V.times(D.times(V.transpose())).getArray();
			inv = uncompressHessian(inv);
			for (int j=0; j<getTotFeatures(); j++) {
				currError = Math.sqrt(inv[j][j]);
				if (Double.isNaN(currError)) {
					tol = tol*10;
					isNaN = true;
					break;
				}
				errorBars[j] = currError;
				isNaN = false;
			}
		}
		nullVectors = 0;
		for (double currEV : absEV) {
			if (currEV<tol)		nullVectors++;
		}
		
		return Array.clone(errorBars);
	}
		
	public void threadPoolShutdown() {
		swPool.shutdown();
		dpPool.shutdown();
		while (!swPool.isShutdown() || !dpPool.isShutdown()) {
			
		}
		return;
	}
	
	public boolean isShape() {
		return isShape;
	}
		
	public boolean isDinuc() {
		return isDinuc;
	}
		
	public boolean isFlank() {
		return (type<4);
	}
		
	public boolean isNSBinding() {
		return isNSBinding;
	}
	
	public boolean isModeRegression() {
		return true;
	}
		
	public int getL() {
		return l;
	}
	
	public int[] getKs() {
		return ks;
	}
	
	public int[][][] getBetaIdx() {
		return betaIdx;
	}
	
	public int getFlankLength() {
		return flankLength;
	}
	
	public int getNCount() {
		return nCount;
	}
		
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
		
	public double getLambda() {
		return lambda;
	}
		
	public long[] getFlankingSequences() {
		return (new long[]{modes.get(0).fShapeFlankingSequence, 
							modes.get(0).rShapeFlankingSequence});
	}
		
	public double likelihoodNormalizer() {
		return 1.0/nCount;
	}
		
	public int getTotFeatures() {
		if (modeRegression) {
			if (isNSBinding) {
				return nModes+1;
			} else {
				return nModes;
			}
		} else {
			return totFeatures;	
		}
	}
		
	public int getNDimensions() {
		if (modeRegression) {
			if (isNSBinding) {
				return nModes+1;
			} else {
				return nModes;
			}
		} else {
			if (nucSymmetries!=null || dinucSymmetries!=null) {
				return nSymmetrizedFeatures();
			} else {
				return totFeatures;
			}
		}
	}
		
	public double[] getNucVector(CompactGradientOutput in, int modeIdx) {
		return Arrays.copyOfRange(betas, betaIdx[modeIdx][1][0], betaIdx[modeIdx][1][1]+1);
	}
	
	public double[] getDinucVector(CompactGradientOutput in, int modeIdx) {
		if (isDinuc) {
			return Arrays.copyOfRange(betas, betaIdx[modeIdx][2][0], betaIdx[modeIdx][2][1]+1);
		} else {
			return null;
		}
	}
	
	public double[] getShapeVector(CompactGradientOutput in, int modeIdx) {
		if (isShape) {
			return Arrays.copyOfRange(betas, betaIdx[modeIdx][3][0], betaIdx[modeIdx][3][1]+1);
		} else {
			return null;
		}
	}
	
	public double getNSGradient(CompactGradientOutput in) {
		double[] gv = in.gradientVector;
		if (isNSBinding) {
			return gv[gv.length-1];
		} else {
			return 0;
		}
	}
	
	public int getNShapeClasses() {
		return nShapeClasses;
	}
	
	public String[] getNucSymmetries() {
		return nucSymmetries;
	}
	
	public String[] getDinucSymmetries() {
		return dinucSymmetries;
	}

	public double getNSBeta() {
		return Math.log(nsBindingValue);
	}
	
	public double[] getNucBetas(int modeIdx) {
		return Array.clone(modes.get(modeIdx).nucBetas);
	}
	
	public double[] getDinucBetas(int modeIdx) {
		if (isDinuc) {
			return Array.clone(modes.get(modeIdx).dinucBetas);
		} else {
			return null;
		}
	}
	
	public double[] getShapeBetas(int modeIdx) {
		if (isShape) {
			return Array.clone(modes.get(modeIdx).shapeBetas);
		} else {
			return null;
		}
	}
	
	public double[] getPositionVector() {
		if (modeRegression) {
			return Array.clone(modeAffinities);
		} else {
			return Array.clone(betas);
		}
	}
	
	public double[] getModeAffinities() {
		return Array.clone(modeAffinities);
	}
	
	public double[] getMergedPositionVector() {
		double[] output;
		
		if (modeRegression) {
			M = featureM;
			output = symmetrizedModeAffinity(betas, modeAffinities);
			M = null;
			return output;
		} else {
			return getPositionVector();
		}
	}
	
	public Shape getShapeModel() {
		return shapeModel;
	}
	
	public double maxLikelihood() {
		double output	= 0;
		for (int i : R1Counts) {
			output -= i*Math.log(i);
		}
		output += nCount*Math.log(nCount);
		return output;
	}
	
	private void threadSchedule(int nThreads) {
		int currIdx;
		int nDataPoints		= R1Probes.length;
		swThreadRange		= new int[nThreads][2];
		dpThreadRange		= new int[nThreads][2];
		int divLength 		= (int) Math.floor( ((double)nDataPoints)/((double) nThreads) );
		int remainder 		= nDataPoints % nThreads;
		
		swThreadRange[0][0] = 0;
		for (int i=0; i<nThreads-1; i++) {
			if (remainder>0) {
				swThreadRange[i][1] = swThreadRange[i][0] + divLength+1;
				remainder--;
			} else {
				swThreadRange[i][1] = swThreadRange[i][0] + divLength;
			}
			swThreadRange[i+1][0] = swThreadRange[i][1];
		}
		swThreadRange[nThreads-1][1] = nDataPoints;
		
		if (nModes < nThreads) {
			currIdx = 0;
			for (int i=0; i<nModes; i++) {
				dpThreadRange[i][0] = currIdx;
				dpThreadRange[i][1] = currIdx+1;
				currIdx++;
			}
		} else {
			divLength			= (int) Math.floor( ((double) nModes)/((double) nThreads));
			remainder			= nModes % nThreads;
			dpThreadRange[0][0] = 0;
			for (int i=0; i<nThreads-1; i++) {
				if (remainder>0) {
					dpThreadRange[i][1] = dpThreadRange[i][0] + divLength+1;
					remainder--;
				} else {
					dpThreadRange[i][1] = dpThreadRange[i][0] + divLength;
				}
				dpThreadRange[i+1][0] = dpThreadRange[i][1];
			}
			dpThreadRange[nThreads-1][1] = nModes;
		}
	}
	
	//Reverse betas for all binding modes
	protected void reverseBetas() {
		for (int i=0; i<nModes; i++) {
			modes.get(i).reverseBetas();
		}
	}

	private void reverseGradients() {
		double[] output		= new double[totFeatures];

		for (int i=0; i<totFeatures; i++) {
			output[i] = gradients[(int) reverseMap[i]];
		}
		gradients = output;
	}
	
	private void reverseHessian() {
		double[][] output	= new double[totFeatures][totFeatures];
		
		for (int i=0; i<totFeatures; i++) {
			for (int j=0; j<totFeatures; j++) {
				output[i][j] = hessian[(int) reverseMap[i]][(int) reverseMap[j]];
			}
		}
		hessian = output;
	}
	
	private void blockDiagMatrix(double[][] start, double[][] diag, int offset) {
		for (int i=0; i<diag.length; i++) {
			for (int j=0; j<diag.length; j++) {
				start[i+offset][j+offset] = diag[i][j];
			}
		}
	}
	
	public void normalForm() {
		//Do nothing as this can screw up fitting
	}
	
//	public void printAllBetas() {
//		for (int i=0; i<nModes; i++) {
//			System.out.print("Mode "+i+":\t");
//			Array.print(modes.get(i).nucBetas);
//		}
//	}
	
	public double[] normalize() {
		double maxVal, shift;
		double[] output = null;
		double[] shifts = (isNSBinding) ? new double[nModes+1] : new double[nModes];
		double[] nucBetas, dinucBetas, shapeBetas;
		
		//Normalize each mode
		for (int i=0; i<nModes; i++) {
			nucBetas	= getNucBetas(i);
			dinucBetas	= getDinucBetas(i);
			shapeBetas	= getShapeBetas(i);
			
			//First normalize nucleotide values
			shift = 0;
			for (int offset=0; offset<modes.get(i).k; offset++) {
				maxVal = nucBetas[offset*4];				//First find maximum value within a block
				for (int idx=1; idx<4; idx++) {
					if (nucBetas[offset*4+idx] > maxVal) {
						maxVal = nucBetas[offset*4+idx];
					}
				}
				for (int idx=0; idx<4; idx++) {			//subtract max value from the whole block
					nucBetas[offset*4+idx] -= maxVal;
				}
				shift -= maxVal;				//accumulate total shift across blocks
			}
			output = Array.cat(output, nucBetas);
			//Next, normalize dinucleotide values
			if (isDinuc) {
				for (int offset=0; offset<(modes.get(i).k-1); offset++) {
					maxVal = dinucBetas[offset*16];		//First find maximum value within a block
					for (int idx=1; idx<16; idx++) {
						if (dinucBetas[offset*16+idx] > maxVal) {
							maxVal = dinucBetas[offset*16+idx];
						}
					}
					for (int idx=0; idx<16; idx++) {		//subtract max value from the whole block
						dinucBetas[offset*16+idx] -= maxVal;
					}
					shift -= maxVal;				//accumulate total shift across blocks
				}
				output = Array.cat(output, dinucBetas);
			}
			output = Array.cat(output, shapeBetas);
			//Record overall shift
			shifts[i] = -shift;
		}
		if (isNSBinding) {
			shifts[nModes] = getNSBeta();
			output = Array.cat(output, 0);
		}
		
		//Compute overall scaling
		shift = Array.max(shifts);
		for (int j=0; j<shifts.length; j++) {
			shifts[j] -= shift;
		}
		
		return symmetrizedModeAffinity(output, shifts);
	}
	
	public MultinomialFit generateFit(double[] seed) {
		return (new MultinomialFit(this, seed));
	}
	
	private void symmetryHelper(String symString, int blockSize, int nBases, 
			ArrayList<double[]> matrix, int endIdx, int offset) {
		double[] work, currRow;
		Matrix[] copyMatrix;
		
		if (symString==null) {
			for (int i=0; i<endIdx; i++) {
				work			= new double[totFeatures];
				work[i+offset]	= 1;
				matrix.add(work);
			}
		} else {
			copyMatrix = mMatrix(parseSymmetryString(symString), blockSize, nBases);
			for (int i=0; i<copyMatrix[0].getArray().length; i++) {
				work	= new double[totFeatures];
				currRow	= copyMatrix[0].getArray()[i];
				for (int j=0; j<currRow.length; j++) {
					work[j+offset] = currRow[j];
				}
				matrix.add(work);
			}
		}
	}
	
	public class ThreadedDPEvaluator implements Callable<Double>{
		private int dpType;
		private int startIdx;
		private int endIdx;
		
		public ThreadedDPEvaluator(int dpType, int startIdx, int endIdx) {
			this.dpType					= dpType;
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}

		@Override
		public Double call() throws Exception {
			double output = 0;
			for (int i=startIdx; i<endIdx; i++) {
				if (dpType==0) {
					output += modes.get(i).Z.recursiveZ();
				} else if (dpType==1) {
					modes.get(i).Z.recursiveGradient();
				} else {
					modes.get(i).Z.recursiveHessian();
				}
			}
			return output;
		}
	}
	
	public class ThreadedFunctionEvaluator implements Callable<Double>{
		private int startIdx;
		private int endIdx;
		
		public ThreadedFunctionEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public Double call() throws Exception {
			double output	= 0, sum = 0;
			double[] kappas = new double[totalFrames];
			
			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					output += R1Counts[i]*Math.log(R1R0Prob[i]*sum);
				}
			}
			return output;
		}
	}

	public class ThreadedGradientEvaluator implements Callable<CompactGradientOutput>{
		private int startIdx;
		private int endIdx;
		
		public ThreadedGradientEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public CompactGradientOutput call() throws Exception {
			double functionValue	= 0, sum = 0, nsGradient = 0;
			double[] kappas			= new double[totalFrames];
			double[] gradients		= new double[totFeatures];

			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotide(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {		
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideDinucleotide(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideShape(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideDinucleotideShape(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideNoFlank(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideDinucleotideNoFlank(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideShapeNoFlank(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {						
						modes.get(j).swGradNucleotideDinucleotideShapeNoFlank(R1Probes[i], 
								R1Counts[i], sum, kappas, gradients);
					}
				}
			}
			if (isNSBinding)	gradients[totFeatures-1] = nsGradient;
			
			return (new CompactGradientOutput(functionValue, gradients));
		}
	}
	
	public class ThreadedModeGradientEvaluator implements Callable<CompactGradientOutput>{
		private int startIdx;
		private int endIdx;
		
		public ThreadedModeGradientEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public CompactGradientOutput call() throws Exception {
			double functionValue	= 0, sum = 0, nsGradient = 0, subSum=0;
			double[] kappas			= new double[totalFrames];
			double[] gradients		= new double[nModes];

			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {		
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotide(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShape(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
					for (int j=0; j<nModes; j++) {
						modes.get(j).swNucleotideDinucleotideShapeNoFlank(R1Probes[i], kappas);
					}
					sum = nsBindingValue;
					for (int j=0; j<totalFrames; j++) {
						sum += kappas[j];
					}
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
					nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
					for (int j=0; j<nModes; j++) {
						subSum		= 0;
						for (int k=kappaIdx[j][0]; k<=kappaIdx[j][1]; k++) {
							subSum	+= kappas[k];
						}
						gradients[j]+= R1Counts[i]*subSum/sum;
					}
				}
			}
			if (isNSBinding)	gradients = Array.cat(gradients, nsGradient);
			
			return (new CompactGradientOutput(functionValue, gradients));
		}
	}
	
	public class ThreadedHessianEvaluator implements Callable<CompactGradientOutput>{
		private int startIdx;
		private int endIdx;
		
		public ThreadedHessianEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public CompactGradientOutput call() throws Exception {
			double functionValue	= 0, sum = 0, nsGradient = 0, nsHessian = 0;
			double[] kappas			= new double[totalFrames];
			double[] gradients		= new double[totFeatures];
			double[] tempGradient	= new double[totFeatures];
			double[][] hessian		= new double[totFeatures][totFeatures];
			double[][] augHessian	= new double[totFeatures][totFeatures];
			
			if (type==0) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotide(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotide(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotide(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotide(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==1) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotide(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotide(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotide(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotide(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==2) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideShape(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideShape(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideShape(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideShape(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==3){
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideShape(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideShape(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideShape(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideShape(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==4){
				if (isNSBinding) {					
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideNoFlank(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideNoFlank(R1Probes[i], 
									R1Counts[i], sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideNoFlank(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideNoFlank(R1Probes[i], 
									R1Counts[i], sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==5) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideNoFlank(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideNoFlank(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==6) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideShapeNoFlank(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideShapeNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideShapeNoFlank(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideShapeNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			} else if(type==7) {
				if (isNSBinding) {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideShapeNoFlank(R1Probes[i], kappas);
						}
						sum = nsBindingValue;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						nsGradient		+= R1Counts[i]*(1/sum-1/Z_FFM);
						nsHessian		+= R1Counts[i]*(1/Z_FFM - 1/sum + nsBindingValue*(1/(sum*sum) - 1/(Z_FFM*Z_FFM)));
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideShapeNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								augHessian[j][nonNSFeatures] -= tempGradient[j]/sum;
								tempGradient[j] = 0;
							}
						}
					}
				} else {
					for (int i=startIdx; i<endIdx; i++) {
						for (int j=0; j<nModes; j++) {
							modes.get(j).swNucleotideDinucleotideShapeNoFlank(R1Probes[i], kappas);
						}
						sum = 0;
						for (int j=0; j<totalFrames; j++) {
							sum += kappas[j];
						}
						functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*sum);
						for (int j=0; j<nModes; j++) {						
							modes.get(j).swHessianNucleotideDinucleotideShapeNoFlank(R1Probes[i], R1Counts[i],
									sum, kappas, tempGradient, hessian);
						}
						//Calculate hessian term 2
						for (int j=0; j<nonNSFeatures; j++) {
							if (activeFeatures[j]==1) {
								gradients[j] += tempGradient[j];
								for (int k=0; k<nonNSFeatures; k++) {
									augHessian[j][k] -= tempGradient[j]*tempGradient[k]/R1Counts[i]*activeFeatures[k];
								}
								tempGradient[j] = 0;
							}
						}
					}
				}
			}
			if (isNSBinding) {
				gradients[totFeatures-1] = nsGradient;
				hessian[nonNSFeatures][nonNSFeatures] = nsHessian;
			}
			//Symmetrize Augmented Hesisan and then add
			for (int j=0; j<totFeatures; j++) {
				for (int k=j+1; k<totFeatures; k++) {
					augHessian[k][j] = augHessian[j][k];
				}
			}
			for (int j=0; j<totFeatures; j++) {
				for (int k=0; k<totFeatures; k++) {
					hessian[j][k] += augHessian[j][k];
				}
			}
			
			return (new CompactGradientOutput(functionValue, gradients, hessian));
		}
	}	
}