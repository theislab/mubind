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
import Jama.QRDecomposition;
import dynamicprogramming.*;
import base.*;

public class MultinomialModel extends Model{
	private boolean isDinuc, isShape, isNSBinding, suppressZ, isMemSafe;
	private int l, k, nCount, nThreads, totFeatures, flankLength, nShapeClasses;
	private int nonNSFeatures, type, maxFrames, nucOffset, dinucOffset, shapeOffset;
	private long maxFrameValue, frameMask, shapeMask, fFlankingSequence;
	private long rFlankingSequence, fShapeFlankingSequence, rShapeFlankingSequence;
	private String lFlank, rFlank, nucSymmetry = null, dinucSymmetry = null;
	private double Z_FFM;					//Z of the full model, with NS Binding
	private double nsBindingValue;			//THIS IS THE BETA_NS REPRESENTATION
	private double nsBindingGradient;
	private double lambda = 0;				//L2 Regularization Parameter
	private int[] revHessianIdxMap;
	private double[] nucBetas, nucGradients, dinucBetas;
	private double[] dinucGradients, shapeBetas, shapeGradients;
	private int[][] threadRange;
	private double[][] shapeFeatures;
	private int[] R1Counts;
	private long[] R1Probes;
	private double[] R1R0Prob;
	private Data data;
	private Shape shapeModel;
	private Round0Model R0Model;
	private DynamicProgramming Z;
	private ExecutorService pool;

	public MultinomialModel(int nThreads, Shape shapeModel, Data data, int k, 
			boolean isFlank, int flankLength, boolean isDinuc, boolean isShape, 
			boolean isNSBinding, String nucSymmetry, String dinucSymmetry, boolean suppressZ) {		
		if (!isFlank || flankLength==0) {
			flankLength = 0;
			isFlank = false;
		}
		
		//Get Round0 Information
		R0Model	 			= data.R0Model;
		//Load Shape Table Information
		if (shapeModel!=null) {
			this.shapeModel	= shapeModel;
			shapeFeatures 	= shapeModel.getFeatures();
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
		this.k				= k;
		this.flankLength	= flankLength;
		lFlank				= data.leftFlank;
		rFlank				= data.rightFlank;
		this.nThreads		= nThreads;
		this.isDinuc		= isDinuc;
		this.isShape		= isShape;
		this.isNSBinding	= isNSBinding;
		this.nucSymmetry	= nucSymmetry;
		this.dinucSymmetry	= dinucSymmetry;
		this.suppressZ		= suppressZ;
		constructorHelper(isFlank, R0Model.isFlank(), suppressZ);
		setParams(new double[totFeatures]);
	}
	
	public MultinomialModel(int nThreads, Shape shapeModel, Data data, 
			Round0Model R0Model, int fitIndex, MultinomialResults results, boolean suppressZ) {
		MultinomialFit fit	= results.getFit(fitIndex);

		if (fit.isFlank) {
			flankLength		= fit.flankLength;
		} else {
			flankLength		= 0;
		}
		isDinuc				= fit.isDinuc;
		isNSBinding			= fit.isNSBinding;
		if (fit.isShape) {
			isShape			= true;
			this.shapeModel	= shapeModel;
			this.shapeModel.setFeatures(fit.shapes);
			shapeFeatures	= shapeModel.getFeatures();
			nShapeClasses	= shapeModel.nShapeFeatures();
		}
		l					= results.l;
		k					= fit.k;
		lambda				= fit.lambda;
		nucSymmetry			= fit.nucSymmetry;
		dinucSymmetry		= fit.dinucSymmetry;
		if (data!=null) {
			this.data		= data;
			nCount			= data.nCount;
			R1Counts		= data.counts;
			R1Probes		= data.probes;
			R1R0Prob		= data.R0Prob;
			this.R0Model	= data.R0Model;
			this.nThreads	= nThreads;
			if (nThreads!=0) {
				pool = Executors.newFixedThreadPool(nThreads);
				threadSchedule(nThreads);
			}
			lFlank			= data.leftFlank;
			rFlank			= data.rightFlank;
		} else if (R0Model!=null){
			this.R0Model	= R0Model;
			this.nThreads	= 0;
			lFlank			= results.lFlank;
			rFlank			= results.rFlank;
		} else {
			throw new IllegalArgumentException("Must provide Data OR Round0Model");
		}
		
		//Load Shape Table Information
		if (shapeModel!=null) {
			this.shapeModel	= shapeModel;
			shapeFeatures 	= shapeModel.getFeatures();
			nShapeClasses	= shapeModel.nShapeFeatures();	
		}
		//Load Parameters
		constructorHelper(fit.isFlank, this.R0Model.isFlank(), suppressZ);
		setParams(fit.positionVector());
	}
	
	private void constructorHelper(boolean isFlank, boolean isR0Flank, boolean suppressZ) {
		int currOffset = 0;
		double[] work;
		ArrayList<double[]> matrix = new ArrayList<double[]>();
		
		//Create and Schedule Threads
		if (nThreads!=0) {
			pool = Executors.newFixedThreadPool(nThreads);
			threadSchedule(nThreads);
		}
		//Runtime Parameters
		maxFrames			= (isFlank) ? l-k+1+2*flankLength : l-k+1;
		maxFrameValue		= (long) Math.pow(4, k);
		frameMask			= maxFrameValue-1;
		shapeMask			= (long) Math.pow(4, k+4)-1;
		nucOffset			= 4*k;
		dinucOffset			= 4*k+16*(k-1);
		shapeOffset			= (isDinuc) ? dinucOffset+nShapeClasses*k : nucOffset+nShapeClasses*k;
		long tempLeft		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
		long tempRight		= (new Sequence(rFlank, 0, rFlank.length())).getValue();
		long tempLeftRC		= reverseComplement(tempRight, rFlank.length());
		long tempRightRC	= reverseComplement(tempLeft, lFlank.length());
		tempRight			= reverse(tempRight, rFlank.length());
		tempRightRC			= reverse(tempRightRC, lFlank.length());
		if (isFlank) {
			long flankMask			= (long) Math.pow(4, flankLength) - 1;
			long shapeFlankMask		= (long) Math.pow(4, flankLength+2) - 1;
			fFlankingSequence		= ((tempLeft & flankMask) << 2*l);
			fFlankingSequence		<<= 2*flankLength;
			fFlankingSequence		= fFlankingSequence | reverse((tempRight & flankMask), flankLength);
			rFlankingSequence		= ((tempLeftRC & flankMask) << 2*l);
			rFlankingSequence		<<= 2*flankLength;
			rFlankingSequence		= rFlankingSequence | reverse((tempRightRC & flankMask), flankLength);
			fShapeFlankingSequence	= ((tempLeft & shapeFlankMask)) << 2*l;
			fShapeFlankingSequence	<<= 2*(flankLength+2);
			fShapeFlankingSequence	= fShapeFlankingSequence | reverse((tempRight & shapeFlankMask), flankLength+2);
			rShapeFlankingSequence	= ((tempLeftRC & shapeFlankMask) << 2*l);
			rShapeFlankingSequence	<<= 2*(flankLength+2);
			rShapeFlankingSequence	= rShapeFlankingSequence | reverse((tempRightRC & shapeFlankMask), flankLength+2);	
		} else {
			fShapeFlankingSequence	= ((tempLeft & 15) << 2*l) << 4;
			fShapeFlankingSequence	= fShapeFlankingSequence | reverse((tempRight & 15), 2);
			rShapeFlankingSequence	= ((tempLeftRC & 15) << 2*l) << 4;
			rShapeFlankingSequence	= rShapeFlankingSequence | reverse((tempRightRC & 15), 2);
		}
		
		totFeatures			+= 4*k;
		if (isDinuc)		totFeatures += 16*(k-1);
		if (isShape)		totFeatures += nShapeClasses*k;
		nonNSFeatures		= totFeatures;
		if (isNSBinding)	totFeatures++;
		
		//Build symmetry matrices 
		symmetryHelper(nucSymmetry, 4, 1, matrix, 4*k, currOffset);
		currOffset += 4*k;
		if (isDinuc) {	
			symmetryHelper(dinucSymmetry, 16, 2, matrix, 16*(k-1), currOffset);
			currOffset += 16*(k-1);
		}
		if (isShape) {
			symmetryHelper(nucSymmetry, nShapeClasses, 0, matrix, nShapeClasses*k, currOffset);
		}
		if (isNSBinding) {
			work = new double[totFeatures];
			work[totFeatures-1] = 1;
			matrix.add(work);
		}
		M = matrixBuilder(matrix, totFeatures);
		
		//create mapping for inverting hessian
		revHessianIdxMap	= new int[4*k];		
		for (int i=0; i<4*k; i++) {
			revHessianIdxMap[i] = i;
		}
		revHessianIdxMap = Array.blockReverse(revHessianIdxMap, 4);
		if (isDinuc) {		//do the same for dinuc features
			int[] dinucIdxMap	= new int[16*(k-1)];
			for (int i=0; i<16*(k-1); i++) {
				dinucIdxMap[i] = i+4*k;
			}
			dinucIdxMap = Array.blockReverse(dinucIdxMap, 16);
			revHessianIdxMap = Array.cat(revHessianIdxMap, dinucIdxMap);
		}
		if (isShape) {		//do the same for shape features
			int offset			= (isDinuc) ? dinucOffset : nucOffset;
			int[] shapeIdxMap	= new int[nShapeClasses*k];
			for (int i=0; i<nShapeClasses*k; i++) {
				shapeIdxMap[i] = i+offset;
			}
			shapeIdxMap = Array.blockReverse(shapeIdxMap, nShapeClasses);
			revHessianIdxMap = Array.cat(revHessianIdxMap, shapeIdxMap);			
		}
		//NSBinding
		if (isNSBinding) {
			revHessianIdxMap = Array.cat(revHessianIdxMap, new int[]{revHessianIdxMap.length});
		}
				
		//What type of sliding window should be used?
		if (isR0Flank) {
			if (isFlank) {
				if (isDinuc) {
					if (isShape) {
						type = 3;		//Nuc+Dinuc+Shape
						if (!suppressZ) {
							Z = new FullFeatureNucleotideDinucleotideShape(l, k, isNSBinding, flankLength, lFlank, rFlank, R0Model, shapeModel);							
						}
					} else {
						type = 1;		//Nuc+Dinuc
						if (!suppressZ) {
							Z = new FullFeatureNucleotideDinucleotide(l, k, isNSBinding, flankLength, lFlank, rFlank, R0Model);							
						}
					}
				} else {
					if (isShape) {
						type = 2;		//Nuc+Shape
						if (!suppressZ) {
							Z = new FullFeatureNucleotideShape(l, k, isNSBinding, flankLength, lFlank, rFlank, R0Model, shapeModel);							
						}
					} else {
						type = 0;		//Nuc
						if (!suppressZ) {
							Z = new FullFeatureNucleotide(l, k, isNSBinding, flankLength, lFlank, rFlank, R0Model);							
						}
					}
				}			
			} else {
				if (isDinuc) {
					if (isShape) {
						type = 7;		//Nuc+Dinuc+Shape
						if (!suppressZ) {
							Z = new FullFeatureNucleotideDinucleotideShapeNoFlank(l, k, isNSBinding, lFlank, rFlank, R0Model, shapeModel);							
						}
					} else {
						type = 5;		//Nuc+Dinuc
						if (!suppressZ) {
							Z = new FullFeatureNucleotideDinucleotideNoFlank(l, k, isNSBinding, lFlank, rFlank, R0Model);							
						}
					}
				} else {
					if (isShape) {
						type = 6;		//Nuc+Shape
						if (!suppressZ) {
							Z = new FullFeatureNucleotideShapeNoFlank(l, k, isNSBinding, lFlank, rFlank, R0Model, shapeModel);							
						}
					} else {
						type = 4;		//Nuc
						if (!suppressZ) {
							Z = new FullFeatureNucleotideNoFlank(l, k, isNSBinding, lFlank, rFlank, R0Model);							
						}
					}
				}
			}	
		}
		
		//Ensure that a hessian calculation is 'memory safe' (less than 5GB)
		isMemSafe = (totFeatures*totFeatures*maxFrames*
				Math.pow(4, R0Model.getK()-1)*8*4<=5E9);
	}
	
	@Override
	public MultinomialModel clone() {
		boolean isFlank			= (flankLength>0);
		MultinomialModel output = new MultinomialModel(nThreads, shapeModel, 
				data, k, isFlank, flankLength, isDinuc, isShape, 
				isNSBinding, nucSymmetry, dinucSymmetry, suppressZ);
		output.setParams(nucBetas, dinucBetas, shapeBetas, Math.log(nsBindingValue));
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
		double[] output; 
		
		if (shiftPositions<0) {				//shift left
			output = Array.cycleLeft(Arrays.copyOfRange(originalBetas, 0, nucOffset), -shiftPositions*4);
			if (isDinuc) {
				output = Array.cat(output, Array.cycleLeft(
						Arrays.copyOfRange(originalBetas, nucOffset, dinucOffset), -shiftPositions*16));
				if (isShape) {
					output = Array.cat(output, Array.cycleLeft(
						Arrays.copyOfRange(originalBetas, dinucOffset, shapeOffset), -shiftPositions*nShapeClasses));
				}
			} else if (isShape) {
				output = Array.cat(output, Array.cycleLeft(
						Arrays.copyOfRange(originalBetas, nucOffset, shapeOffset), -shiftPositions*nShapeClasses));
			}
		} else {
			output = Array.cycleRight(Arrays.copyOfRange(originalBetas, 0, nucOffset), shiftPositions*4);
			if (isDinuc) {
				output = Array.cat(output, Array.cycleRight(
						Arrays.copyOfRange(originalBetas, nucOffset, dinucOffset), shiftPositions*16));
				if (isShape) {
					output = Array.cat(output, Array.cycleRight(
						Arrays.copyOfRange(originalBetas, dinucOffset, shapeOffset), shiftPositions*nShapeClasses));
				}
			} else if (isShape) {
				output = Array.cat(output, Array.cycleRight(
						Arrays.copyOfRange(originalBetas, nucOffset, shapeOffset), shiftPositions*nShapeClasses));
			}
		}
		if (isNSBinding) {
			output = Array.cat(output, new double[]{originalBetas[originalBetas.length-1]});
		}
		return output;
	}
	
	public double[] orthogonalStep(double[] currPos, int position, double stepSize) {
		int offset;
		double normalizer	= (isNSBinding) ? 1.0 : 0.0;
		double[] output		= Array.clone(currPos);
		
		if (position<4*k) {								//Nucleotide region
			offset = position/4;
			normalizer += 4;
			for (int i=0; i<4; i++) {
				output[offset*4+i] -= stepSize/normalizer;
			}
			if (isNSBinding)	output[output.length-1] -= stepSize/normalizer;
		} else if(isDinuc && (position<dinucOffset)) {	//Dinuc Region
			offset = (position-4*k)/16;
			normalizer += 16;
			for (int i=0; i<16; i++) {
				output[4*k+offset*16+i] -= stepSize/normalizer;
			}
			if (isNSBinding)	output[output.length-1] -= stepSize/normalizer;
		}		
		output[position] += stepSize;					//Always move one unit in the required direction
		
		return symmetrize(output);						//Symmetrize!
	}
	
	public void setParams(double[] nucBetas, double[] dinucBetas, double[] shapeBetas, double nsBinding) {
		double[] dinucAlphas = null;
		
		this.nucBetas	= nucBetas;
		if (isDinuc) {
			this.dinucBetas	= dinucBetas;
			dinucAlphas		= Array.exp(dinucBetas);
		}
		if (isShape)	this.shapeBetas	= shapeBetas;
		if (isNSBinding) {
			nsBindingValue	= Math.exp(nsBinding);			//nsBinding is the ENERGY value
		} else {
			nsBindingValue	= 0;
		}
		if (!suppressZ) {
			Z.setAlphas(Array.exp(nucBetas), dinucAlphas, shapeBetas);			
		}
	}
	
	//setParams does NOT SYMMETRIZE input
	public void setParams(double[] input) {
		double[] nucBetas = Arrays.copyOfRange(input, 0, nucOffset);
		double[] dinucBetas, shapeBetas;
		if (isDinuc) {
			dinucBetas = Arrays.copyOfRange(input, nucOffset, dinucOffset);
			shapeBetas = (isShape) ? Arrays.copyOfRange(input, dinucOffset, shapeOffset) : null;
		} else {
			dinucBetas = null;
			shapeBetas = (isShape) ? Arrays.copyOfRange(input, nucOffset, shapeOffset) : null;
		}
		double nsBinding = (isNSBinding) ? input[input.length-1] : 0;
		setParams(nucBetas, dinucBetas, shapeBetas, nsBinding);
	}
	
	private void resetGradients() {
		nsBindingGradient	= 0;
		nucGradients		= new double[nucBetas.length];
		dinucGradients		= (isDinuc) ? new double[dinucBetas.length] : null;
		shapeGradients		= (isShape) ? new double[shapeBetas.length] : null;
	}
	
	private void resetHessian() {
		resetGradients();
		hessian = new double[totFeatures][totFeatures];
	}
		
	public double getZ() {
		return (Z.recursiveZ()/R0Model.getZ() + nsBindingValue);
	}
		
	public double functionEval() throws InterruptedException, ExecutionException {
		Z_FFM			= Z.recursiveZ()/R0Model.getZ() + nsBindingValue;
		double output	= nCount*Math.log(Z_FFM);
		List<Future<Double>> threadOutput;
		Set<Callable<Double>> tasks = new HashSet<Callable<Double>>(nThreads);
		
		reverseBetas();			//Need the proper orientation for sliding windows
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			tasks.add(new ThreadedFunctionEvaluator(threadRange[i][0], threadRange[i][1]));
		}
		//Launch Threads
		threadOutput = pool.invokeAll(tasks);
		//Sum up value and return
		for (Future<Double> currentThread : threadOutput) {
			output -= currentThread.get();
		}
		reverseBetas();			//Return to the original state
		return output + lambda*Math.pow(Array.norm(getPositionVector()),2);
	}

	public CompactGradientOutput getGradient() {
		return new CompactGradientOutput(0, toVector(isNSBinding, nsBindingGradient, 
				nucGradients, dinucGradients, shapeGradients));
	}
		
	public CompactGradientOutput gradientEval() throws InterruptedException, ExecutionException {
		double functionValue;
		List<Future<CompactGradientOutput>> threadOutput;
		Set<Callable<CompactGradientOutput>> tasks = new HashSet<Callable<CompactGradientOutput>>(nThreads);
		CompactGradientOutput threadResult;
		
		resetGradients();		//Clear out values for old gradients
		Z.recursiveGradient();	//Calculate gradient of Z
		Z_FFM				= Z.getZ()/R0Model.getZ() + nsBindingValue;
		functionValue		= nCount*Math.log(Z_FFM);
		nucGradients		= Array.scalarMultiply(Z.getNucGradients(), nCount/(Z_FFM*R0Model.getZ()));
		if (isDinuc)		dinucGradients	= Array.scalarMultiply(Z.getDinucGradients(), nCount/(Z_FFM*R0Model.getZ()));
		if (isShape)		shapeGradients	= Array.scalarMultiply(Z.getShapeGradients(), nCount/(Z_FFM*R0Model.getZ()));
		
		reverseBetas();			//Need the proper orientation for sliding windows
		reverseGradients();
		//Assign Threads
		for (int i=0; i<nThreads; i++) {
			tasks.add(new ThreadedGradientEvaluator(threadRange[i][0], threadRange[i][1]));
		}
		//Launch Threads
		threadOutput = pool.invokeAll(tasks);
		//Sum up value and return
		for (Future<CompactGradientOutput> currentThread : threadOutput) {
			threadResult		= currentThread.get();
			functionValue		-= threadResult.functionValue;
			nsBindingGradient	-= getNSGradient(threadResult);
			nucGradients		= Array.subtract(nucGradients, getNucVector(threadResult));
			if (isDinuc)		dinucGradients	= Array.subtract(dinucGradients, getDinucVector(threadResult));
			if (isShape)		shapeGradients	= Array.subtract(shapeGradients, getShapeVector(threadResult));
		}
		reverseBetas();			//Return to the original state
		reverseGradients();
		nsBindingGradient	= nsBindingValue*nsBindingGradient + 2*lambda*getNSBeta();
		
		functionValue	+= lambda*Math.pow(Array.norm(getPositionVector()),2);
		nucGradients	= Array.addScalarMultiply(nucGradients, 2*lambda, nucBetas);
		if (isDinuc)	dinucGradients = Array.addScalarMultiply(dinucGradients, 2*lambda, dinucBetas);
		if (isShape)	shapeGradients = Array.addScalarMultiply(shapeGradients, 2*lambda, shapeBetas);
		
		return (new CompactGradientOutput(functionValue, toVector(isNSBinding, 
				nsBindingGradient, nucGradients, dinucGradients, shapeGradients)));
	}
	
	public CompactGradientOutput hessianEval() throws InterruptedException, ExecutionException {
		//Ensure 'memory safe' operation
		if (isMemSafe) {
			double functionValue;
			double[] zGradient;
			List<Future<CompactGradientOutput>> threadOutput;
			Set<Callable<CompactGradientOutput>> tasks = new HashSet<Callable<CompactGradientOutput>>(nThreads);
			CompactGradientOutput threadResult;
			
			resetHessian();
			Z.recursiveHessian();
			Z_FFM			= Z.getZ()/R0Model.getZ() + nsBindingValue;
			functionValue	= nCount*Math.log(Z_FFM);
			nucGradients	= Array.scalarMultiply(Z.getNucGradients(), 1/R0Model.getZ());					//dZ/dBeta
			if (isDinuc)	dinucGradients	= Array.scalarMultiply(Z.getDinucGradients(), 1/R0Model.getZ());
			if (isShape)	shapeGradients	= Array.scalarMultiply(Z.getShapeGradients(), 1/R0Model.getZ());
			hessian			= Array.scalarMultiply(Z.getHessian(), nCount/(Z_FFM*R0Model.getZ()));					//dZ/dBeta1*dBeta2
			
			//subtract n/Z^2 dZ/dBeta1*dZ/dBeta2
			//create dZ/dBeta
			zGradient		= Array.clone(nucGradients);
			if (dinucGradients != null)	{
				this.dinucGradients	= Array.clone(dinucGradients);
				zGradient	= Array.cat(zGradient, dinucGradients);
			}
			if (shapeGradients != null)	{
				this.shapeGradients	= Array.clone(shapeGradients);
				zGradient	= Array.cat(zGradient, shapeGradients);
			}
			//subtract
			for (int i=0; i<nonNSFeatures; i++) {
				for (int j=0; j<nonNSFeatures; j++) {
					hessian[i][j] -= nCount/(Z_FFM*Z_FFM)*zGradient[i]*zGradient[j];
				}
				//nsBinding offset
				if (isNSBinding) {
					hessian[i][nonNSFeatures] -= nCount/(Z_FFM*Z_FFM)*zGradient[i];
					hessian[nonNSFeatures][i] -= nCount/(Z_FFM*Z_FFM)*zGradient[i];				
				}
			}
			//create proper offset for gradient evaluation
			nucGradients	= Array.scalarMultiply(nucGradients, nCount/Z_FFM);
			if (isDinuc)	dinucGradients	= Array.scalarMultiply(dinucGradients, nCount/Z_FFM);
			if (isShape)	shapeGradients	= Array.scalarMultiply(shapeGradients, nCount/Z_FFM);
			
			reverseBetas();			//Need the proper orientation for sliding windows
			reverseGradients();
			reverseHessian();
			//Assign Threads
			for (int i=0; i<nThreads; i++) {
				tasks.add(new ThreadedHessianEvaluator(threadRange[i][0], threadRange[i][1]));
			}
			//Launch Threads
			threadOutput = pool.invokeAll(tasks);
			//Sum up value and return
			for (Future<CompactGradientOutput> currentThread : threadOutput) {
				threadResult		= currentThread.get();
				functionValue		-= threadResult.functionValue;
				nsBindingGradient	-= getNSGradient(threadResult);
				nucGradients		= Array.subtract(nucGradients, getNucVector(threadResult));
				if (isDinuc)		dinucGradients	= Array.subtract(dinucGradients, getDinucVector(threadResult));
				if (isShape)		shapeGradients	= Array.subtract(shapeGradients, getShapeVector(threadResult));
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
			
			nsBindingGradient	= nsBindingValue*nsBindingGradient + 2*lambda*getNSBeta();
			if (isNSBinding) {
				for (int i=0; i<nonNSFeatures; i++) {
					hessian[i][nonNSFeatures] *= nsBindingValue;
					hessian[nonNSFeatures][i] *= nsBindingValue;
				}
				hessian[nonNSFeatures][nonNSFeatures] *= nsBindingValue;
			}
			
			for (int i=0; i<totFeatures; i++) {
				hessian[i][i] += 2*lambda;
			}
			
			functionValue	+= lambda*Math.pow(Array.norm(getPositionVector()),2);
			nucGradients	= Array.addScalarMultiply(nucGradients, 2*lambda, nucBetas);
			if (isDinuc)	dinucGradients = Array.addScalarMultiply(dinucGradients, 2*lambda, dinucBetas);
			if (isShape)	shapeGradients = Array.addScalarMultiply(shapeGradients, 2*lambda, shapeBetas);
			
			return (new CompactGradientOutput(functionValue, toVector(isNSBinding, nsBindingGradient, nucGradients, dinucGradients, shapeGradients), hessian));
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
		pool.shutdown();
		while (!pool.isShutdown()) {
			
		}
		return;
	}
	
	public double seqEval(long input) {
		double output = 0;
		reverseBetas();
		switch (type) {
			case 0:		output = swNucleotide(input);
						break;
			case 1:		output = swNucleotideDinucleotide(input);
						break;
			case 2:		output = swNucleotideShape(input);
						break;
			case 3:		output = swNucleotideDinucleotideShape(input);
						break;
			case 4:		output = swNucleotideNoFlank(input);
						break;
			case 5:		output = swNucleotideDinucleotideNoFlank(input);
						break;
			case 6: 	output = swNucleotideShapeNoFlank(input);
						break;
			case 7:		output = swNucleotideDinucleotideShapeNoFlank(input);
		}
		reverseBetas();
		return output;
	}
	
	public double[] seqEval(long[] input) {
		double[] output = new double[input.length];
		reverseBetas();
		if (type==0) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotide(input[i]);
			}
		} else if(type==1) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideDinucleotide(input[i]);
			}
		} else if(type==2) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideShape(input[i]);
			}
		} else if(type==3) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideDinucleotideShape(input[i]);
			}
		} else if(type==4){
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideNoFlank(input[i]);
			}
		} else if(type==5) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideDinucleotideNoFlank(input[i]);
			}
		} else if(type==6) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideShapeNoFlank(input[i]);
			}
		} else if(type==7) {
			for (int i=0; i<input.length; i++) {
				output[i] = swNucleotideDinucleotideShapeNoFlank(input[i]);
			}
		} 
		reverseBetas();
		return output;
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
	
	public int getL() {
		return l;
	}
	
	public int getK() {
		return k;
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
		return (new long[]{fShapeFlankingSequence, rShapeFlankingSequence});
	}
	
	public double likelihoodNormalizer() {
		return 1.0/nCount;
	}
	
	public int getTotFeatures() {
		return totFeatures;
	}
	
	public int getNDimensions() {
		if (nucSymmetry!=null || dinucSymmetry!=null) {
			return nSymmetrizedFeatures();
		} else {
			return totFeatures;
		}
	}
	
	public double[] toVector(boolean isNSBinding, double nsGradient, double[] 
			nucGradients, double[] dinucGradients, double[] shapeGradients) {
		double[] gradientVector		= Array.clone(nucGradients);
		if (dinucGradients != null)	{
			gradientVector	= Array.cat(gradientVector, dinucGradients);
		}
		if (shapeGradients != null)	{
			gradientVector	= Array.cat(gradientVector, shapeGradients);
		}
		if (isNSBinding) {
			gradientVector	= Array.cat(gradientVector, new double[]{nsGradient});
		}
		return gradientVector;
	}
	
	public double[] getNucVector(CompactGradientOutput in) {
		return Arrays.copyOfRange(in.gradientVector, 0, nucOffset);
	}
	
	public double[] getDinucVector(CompactGradientOutput in) {
		if (isDinuc) {
			return Arrays.copyOfRange(in.gradientVector, nucOffset, dinucOffset);
		} else {
			return null;
		}
	}
	
	public double[] getShapeVector(CompactGradientOutput in) {
		if (isDinuc) {
			if (isShape) {
				return Arrays.copyOfRange(in.gradientVector, dinucOffset, shapeOffset);
			} else {
				return null;
			}
		} if (isShape) {
			return Arrays.copyOfRange(in.gradientVector, nucOffset, shapeOffset);
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
	
	public String getNucSymmetry() {
		return nucSymmetry;
	}
		
	public String getDinucSymmetry() {
		return dinucSymmetry;
	}

	public double getNSBeta() {
		return Math.log(nsBindingValue);
	}
	
	public double[] getNucBetas() {
		return Array.clone(nucBetas);
	}
	
	public double[] getDinucBetas() {
		if (isDinuc) {
			return Array.clone(dinucBetas);
		} else {
			return null;
		}
	}
	
	public double[] getShapeBetas() {
		if (isShape) {
			return Array.clone(shapeBetas);
		} else {
			return null;
		}
	}
	
	public double[] getPositionVector() {
		double[] output	= Array.clone(nucBetas);
		
		if (isDinuc)	output = Array.cat(output, dinucBetas);
		if (isShape)	output = Array.cat(output, shapeBetas);
		if (isNSBinding)output = Array.cat(output, new double[]{this.getNSBeta()});
		
		return output;
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
	
	public Object[] swRanker(long input) {
		double[] totalSum = new double[maxFrames*2];
		double weight;
		double fSubSum	= 0;
		double rSubSum	= 0;
		int fShapeIdx	= 0;
		int rShapeIdx	= 0;
		long fSubString;
		long rSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fShapeSeq = fShapeFlankingSequence | (input << 2*(flankLength+2));
		long rShapeSeq = rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long fShapeSubString;
		long rShapeSubString;
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				if (isDinuc) {
					fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
					rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				}
				if (isShape) {
					fShapeIdx = (int) (fShapeSubString & 1023);
					rShapeIdx = (int) (rShapeSubString & 1023);
					for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
						fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
						rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
					}
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			if (isShape) {
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
			}
			totalSum[j] 			= fSubSum;
			totalSum[j+maxFrames]	= rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum	= Array.exp(totalSum);
		weight		= R0Model.getSeqProb(input)*(Array.sum(totalSum)+nsBindingValue);
		return (new Object[]{totalSum, weight});
	}
	
	private double swNucleotide(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		long forwardSubString;
		long reverseSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotide(long input, int ki, double[] nucGrads) {
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long forwardSubString;
		long reverseSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum = totalSum+nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				nucGrads[loc*4+((int) (forwardSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (reverseSubString&3))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotide(long input, int ki, double[] nucGrads, double[][] hessian) {
		int fwdIdx, revIdx;
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long forwardSubString;
		long reverseSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long fwdBinSubString;
		long reverseStrand		= revStrand;
		long revBinSubString;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums) + nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdIdx = loc*4+((int) (forwardSubString&3));
				revIdx = loc*4+((int) (reverseSubString&3));
				for (int subLoc=0; subLoc<k; subLoc++) {
					hessian[fwdIdx][subLoc*4+((int) (fwdBinSubString&3))] += fSubSum;
					hessian[revIdx][subLoc*4+((int) (revBinSubString&3))] += rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				tempGrad[fwdIdx] += fSubSum;
				tempGrad[revIdx] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			nucGrads[i] += tempGrad[i];
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private double swNucleotideNoFlank(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		long forwardSubString;
		long reverseSubString;
		long forwardStrand = input;
		long reverseStrand = reverseComplement(input, l);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideNoFlank(long input, int ki, double[] nucGrads) {
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long forwardSubString;
		long reverseSubString;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum = totalSum+nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				nucGrads[loc*4+((int) (forwardSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (reverseSubString&3))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideNoFlank(long input, int ki, double[] nucGrads, double[][] hessian) {
		int fwdIdx, revIdx;
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long forwardSubString;
		long fwdBinSubString;
		long reverseSubString;
		long revBinSubString;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums) + nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdIdx = loc*4+((int) (forwardSubString&3));
				revIdx = loc*4+((int) (reverseSubString&3));
				for (int subLoc=0; subLoc<k; subLoc++) {
					hessian[fwdIdx][subLoc*4+((int) (fwdBinSubString&3))] += fSubSum;
					hessian[revIdx][subLoc*4+((int) (revBinSubString&3))] += rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				tempGrad[fwdIdx] += fSubSum;
				tempGrad[revIdx] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0)	continue;
			nucGrads[i] += tempGrad[i];
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0)	continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;// - ki*zGradient[i];
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;// - ki*zGradient[i];
		}
		return totalSum;
	}
	
	private double swNucleotideDinucleotide(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		long forwardSubString;
		long reverseSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (reverseSubString & 15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideDinucleotide(long input, int ki, double[] nucGrads, double[] dinucGrads) {
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long forwardSubString;
		long reverseSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (reverseSubString & 15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum = totalSum+nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				nucGrads[loc*4+((int) (forwardSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (reverseSubString&3))] += rSubSum;
				dinucGrads[loc*16+((int) (forwardSubString&15))] += fSubSum;
				dinucGrads[loc*16+((int) (reverseSubString&15))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			nucGrads[(k-1)*4+((int) (forwardSubString&3))] += fSubSum;
			nucGrads[(k-1)*4+((int) (reverseSubString&3))] += rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideDinucleotide(long input, int ki, double[] nucGrads, double[] dinucGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx;
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long forwardSubString;
		long fwdBinSubString;
		long reverseSubString;
		long revBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (reverseSubString & 15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum = totalSum+nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdNucIdx				= loc*4+((int) (forwardSubString&3));
				revNucIdx				= loc*4+((int) (reverseSubString&3));
				fwdDinucIdx				= loc*16+((int) (forwardSubString&15));
				revDinucIdx				= loc*16+((int) (reverseSubString&15));
				nucGrads[fwdNucIdx]		+= fSubSum;
				nucGrads[revNucIdx]		+= rSubSum;
				dinucGrads[fwdDinucIdx] += fSubSum;
				dinucGrads[revDinucIdx]	+= rSubSum;
				fwdDinucIdx				+= nucOffset;
				revDinucIdx				+= nucOffset;
				tempGrad[fwdNucIdx]		+= fSubSum;
				tempGrad[revNucIdx]		+= rSubSum;
				tempGrad[fwdDinucIdx]	+= fSubSum;
				tempGrad[revDinucIdx]	+= rSubSum;
				for (int subLoc=0; subLoc<k-1; subLoc++) {
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;
					hessian[fwdDinucIdx][subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;
					hessian[revDinucIdx][subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;
					hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				hessian[fwdNucIdx][(k-1)*4+((int) (fwdBinSubString&3))]						+= fSubSum;
				hessian[revNucIdx][(k-1)*4+((int) (revBinSubString&3))]						+= rSubSum;
				hessian[fwdDinucIdx][(k-1)*4+((int) (fwdBinSubString&3))]					+= fSubSum;
				hessian[revDinucIdx][(k-1)*4+((int) (revBinSubString&3))]					+= rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fwdBinSubString		= forwardStrand & frameMask;
			revBinSubString		= reverseStrand & frameMask;
			fwdNucIdx			= (k-1)*4+((int) (forwardSubString&3));
			revNucIdx			= (k-1)*4+((int) (reverseSubString&3));
			nucGrads[fwdNucIdx] += fSubSum;
			nucGrads[revNucIdx] += rSubSum;
			tempGrad[fwdNucIdx]	+= fSubSum;
			tempGrad[revNucIdx] += rSubSum;
			for (int subLoc=0; subLoc<k-1; subLoc++) {
				hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]					+= fSubSum;
				hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]					+= rSubSum;
				hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum;
				hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum;
				fwdBinSubString >>= 2;
			revBinSubString >>= 2;
			}
			hessian[fwdNucIdx][(k-1)*4+((int) (fwdBinSubString&3))]	+= fSubSum;
			hessian[revNucIdx][(k-1)*4+((int) (revBinSubString&3))]	+= rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private double swNucleotideDinucleotideNoFlank(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		long forwardSubString;
		long reverseSubString;
		long forwardStrand = input;
		long reverseStrand = reverseComplement(input, l);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				rSubSum	+= dinucBetas[loc*16 + ((int) (reverseSubString&15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideDinucleotideNoFlank(long input, int ki, double[] nucGrads, double[] dinucGrads) {
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long forwardSubString;
		long reverseSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				rSubSum	+= dinucBetas[loc*16 + ((int) (reverseSubString&15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum = totalSum+nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				nucGrads[loc*4+((int) (forwardSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (reverseSubString&3))] += rSubSum;
				dinucGrads[loc*16+((int) (forwardSubString&15))] += fSubSum;
				dinucGrads[loc*16+((int) (reverseSubString&15))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			nucGrads[(k-1)*4+((int) (forwardSubString&3))] += fSubSum;
			nucGrads[(k-1)*4+((int) (reverseSubString&3))] += rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideDinucleotideNoFlank(long input, int ki, double[] nucGrads, double[] dinucGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx;
		double totalSum			= 0;
		double fSubSum;
		double rSubSum;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long forwardSubString;
		long fwdBinSubString;
		long reverseSubString;
		long revBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate String Sum
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				rSubSum	+= dinucBetas[loc*16 + ((int) (reverseSubString&15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		totalSum = Array.sum(fPartialSums) + Array.sum(rPartialSums) + nsBindingValue;
		forwardStrand = fwdStrand;
		reverseStrand = revStrand;
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdNucIdx				= loc*4+((int) (forwardSubString&3));
				revNucIdx				= loc*4+((int) (reverseSubString&3));
				fwdDinucIdx				= loc*16+((int) (forwardSubString&15));
				revDinucIdx				= loc*16+((int) (reverseSubString&15));
				nucGrads[fwdNucIdx]		+= fSubSum;
				nucGrads[revNucIdx]		+= rSubSum;
				dinucGrads[fwdDinucIdx] += fSubSum;
				dinucGrads[revDinucIdx]	+= rSubSum;
				fwdDinucIdx				+= nucOffset;
				revDinucIdx				+= nucOffset;
				tempGrad[fwdNucIdx]		+= fSubSum;
				tempGrad[revNucIdx]		+= rSubSum;
				tempGrad[fwdDinucIdx]	+= fSubSum;
				tempGrad[revDinucIdx]	+= rSubSum;
				for (int subLoc=0; subLoc<k-1; subLoc++) {
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;
					hessian[fwdDinucIdx][subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;
					hessian[revDinucIdx][subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;
					hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				hessian[fwdNucIdx][(k-1)*4+((int) (fwdBinSubString&3))]						+= fSubSum;
				hessian[revNucIdx][(k-1)*4+((int) (revBinSubString&3))]						+= rSubSum;
				hessian[fwdDinucIdx][(k-1)*4+((int) (fwdBinSubString&3))]					+= fSubSum;
				hessian[revDinucIdx][(k-1)*4+((int) (revBinSubString&3))]					+= rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fwdBinSubString		= forwardStrand & frameMask;
			revBinSubString		= reverseStrand & frameMask;
			fwdNucIdx			= (k-1)*4+((int) (forwardSubString&3));
			revNucIdx			= (k-1)*4+((int) (reverseSubString&3));
			nucGrads[fwdNucIdx] += fSubSum;
			nucGrads[revNucIdx] += rSubSum;
			tempGrad[fwdNucIdx]	+= fSubSum;
			tempGrad[revNucIdx] += rSubSum;
			for (int subLoc=0; subLoc<k-1; subLoc++) {
				hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]					+= fSubSum;
				hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]					+= rSubSum;
				hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum;
				hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum;
				fwdBinSubString >>= 2;
				revBinSubString >>= 2;
			}
			hessian[fwdNucIdx][(k-1)*4+((int) (fwdBinSubString&3))]	+= fSubSum;
			hessian[revNucIdx][(k-1)*4+((int) (revBinSubString&3))]	+= rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private double swNucleotideShape(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		int fShapeIdx	= 0;
		int rShapeIdx	= 0;
		long fSubString;
		long rSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fShapeSeq = fShapeFlankingSequence | (input << 2*(flankLength+2));
		long rShapeSeq = rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long fShapeSubString;
		long rShapeSubString;
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideShape(long input, int ki, double[] nucGrads, double[] shapeGrads) {
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long fSubString;
		long rSubString;
		long fShapeSubString;
		long rShapeSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate String Sum		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				nucGrads[loc*4+((int) (fSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (rSubString&3))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					shapeGrads[loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					shapeGrads[loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideShape(long input, int ki, double[] nucGrads, double[] shapeGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx;
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long fSubString;
		long fwdBinSubString;
		long rSubString;
		long revBinSubString;
		long fShapeSubString;
		long fwdShapeBinSubString;
		long rShapeSubString;
		long revShapeBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate String Sum		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*fPartialSums[j]/totalSum;
			rSubSum			= ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {					//Loop over all positions and calculate the gradient
				fwdBinSubString		= forwardStrand & frameMask;
				revBinSubString		= reverseStrand & frameMask;
				fwdShapeBinSubString= fShapeSeq & shapeMask;
				revShapeBinSubString= rShapeSeq & shapeMask;
				fwdNucIdx			= loc*4+((int) (fSubString&3));
				revNucIdx			= loc*4+((int) (rSubString&3));
				fShapeIdx			= (int) (fShapeSubString & 1023);
				rShapeIdx			= (int) (rShapeSubString & 1023);
				shapePosIdx			= loc*nShapeClasses+nucOffset;
				nucGrads[fwdNucIdx] += fSubSum;
				nucGrads[revNucIdx] += rSubSum;
				tempGrad[fwdNucIdx] += fSubSum;
				tempGrad[revNucIdx] += rSubSum;
				for (int csc=0; csc<nShapeClasses; csc++) {
					shapeGrads[shapePosIdx+csc-nucOffset]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= fSubSum*shapeFeatures[fShapeIdx][csc];
					shapeGrads[shapePosIdx+csc-nucOffset]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+nucOffset;
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;		//nuc on nuc
					for (int csc=0; csc<nShapeClasses; csc++) {											//shape on nuc
						hessian[shapePosIdx+csc][subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private double swNucleotideShapeNoFlank(long input) {
		double totalSum		= 0;
		double fSubSum		= 0;
		double rSubSum		= 0;
		int fShapeIdx		= 0;
		int rShapeIdx		= 0;
		long fSubString;
		long rSubString;
		long forwardStrand 	= input;
		long reverseStrand 	= reverseComplement(input, l);
		long fShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long rShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long fShapeSubString;
		long rShapeSubString;
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideShapeNoFlank(long input, int ki, double[] nucGrads, double[] shapeGrads) {
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long fSubString;
		long rSubString;
		long fShapeSubString;
		long rShapeSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate string sum
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				nucGrads[loc*4+((int) (fSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (rSubString&3))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					shapeGrads[loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					shapeGrads[loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideShapeNoFlank(long input, int ki, double[] nucGrads, double[] shapeGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx;
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long fSubString;
		long fwdBinSubString;
		long rSubString;
		long revBinSubString;
		long fShapeSubString;
		long fwdShapeBinSubString;
		long rShapeSubString;
		long revShapeBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate string sum
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums) + nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {					//Loop over all positions and calculate the gradient
				fwdBinSubString		= forwardStrand & frameMask;
				revBinSubString		= reverseStrand & frameMask;
				fwdShapeBinSubString= fShapeSeq & shapeMask;
				revShapeBinSubString= rShapeSeq & shapeMask;
				fwdNucIdx			= loc*4+((int) (fSubString&3));
				revNucIdx			= loc*4+((int) (rSubString&3));
				fShapeIdx			= (int) (fShapeSubString & 1023);
				rShapeIdx			= (int) (rShapeSubString & 1023);
				shapePosIdx			= loc*nShapeClasses+nucOffset;
				nucGrads[fwdNucIdx] += fSubSum;
				nucGrads[revNucIdx] += rSubSum;
				tempGrad[fwdNucIdx] += fSubSum;
				tempGrad[revNucIdx] += rSubSum;
				for (int csc=0; csc<nShapeClasses; csc++) {
					shapeGrads[shapePosIdx+csc-nucOffset]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= fSubSum*shapeFeatures[fShapeIdx][csc];
					shapeGrads[shapePosIdx+csc-nucOffset]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+nucOffset;
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;		//nuc on nuc
					for (int csc=0; csc<nShapeClasses; csc++) {											//shape on nuc
						hessian[shapePosIdx+csc][subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private double swNucleotideDinucleotideShape(long input) {
		double totalSum = 0;
		double fSubSum	= 0;
		double rSubSum	= 0;
		int fShapeIdx	= 0;
		int rShapeIdx	= 0;
		long fSubString;
		long rSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fShapeSeq = fShapeFlankingSequence | (input << 2*(flankLength+2));
		long rShapeSeq = rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long fShapeSubString;
		long rShapeSubString;
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideDinucleotideShape(long input, int ki, double[] nucGrads, double[] dinucGrads, double[] shapeGrads) {
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long fSubString;
		long rSubString;
		long fShapeSubString;
		long rShapeSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate String Sum		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				nucGrads[loc*4+((int) (fSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (rSubString&3))] += rSubSum;
				dinucGrads[loc*16+((int) (fSubString&15))] += fSubSum;
				dinucGrads[loc*16+((int) (rSubString&15))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					shapeGrads[loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					shapeGrads[loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			nucGrads[(k-1)*4+((int) (fSubString&3))] += fSubSum;
			nucGrads[(k-1)*4+((int) (rSubString&3))] += rSubSum;
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				shapeGrads[(k-1)*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
				shapeGrads[(k-1)*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideDinucleotideShape(long input, int ki, double[] nucGrads, double[] dinucGrads, double[] shapeGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx;
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long fSubString;
		long fwdBinSubString;
		long rSubString;
		long revBinSubString;
		long fShapeSubString;
		long fwdShapeBinSubString;
		long rShapeSubString;
		long revShapeBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate String Sum		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*fPartialSums[j]/totalSum;
			rSubSum			= ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString			= forwardStrand & frameMask;
				revBinSubString			= reverseStrand & frameMask;
				fwdShapeBinSubString	= fShapeSeq & shapeMask;
				revShapeBinSubString	= rShapeSeq & shapeMask;
				fwdNucIdx				= loc*4+((int) (fSubString&3));
				revNucIdx				= loc*4+((int) (rSubString&3));
				fwdDinucIdx				= loc*16+((int) (fSubString&15));
				revDinucIdx				= loc*16+((int) (rSubString&15));
				fShapeIdx				= (int) (fShapeSubString & 1023);
				rShapeIdx				= (int) (rShapeSubString & 1023);				
				shapePosIdx				= loc*nShapeClasses+dinucOffset;
				nucGrads[fwdNucIdx]		+= fSubSum;
				nucGrads[revNucIdx]		+= rSubSum;
				tempGrad[fwdNucIdx]		+= fSubSum;
				tempGrad[revNucIdx]		+= rSubSum;
				if (loc < k-1) {
					dinucGrads[fwdDinucIdx] += fSubSum;
					dinucGrads[revDinucIdx]	+= rSubSum;
					fwdDinucIdx				+= nucOffset;
					revDinucIdx				+= nucOffset;
					tempGrad[fwdDinucIdx]	+= fSubSum;
					tempGrad[revDinucIdx]	+= rSubSum;
				}
				for (int csc=0; csc<nShapeClasses; csc++) {
					shapeGrads[shapePosIdx+csc-dinucOffset]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= fSubSum*shapeFeatures[fShapeIdx][csc];
					shapeGrads[shapePosIdx+csc-dinucOffset]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+dinucOffset;
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;		//nuc on nuc
					if (loc < k-1) {
						hessian[fwdDinucIdx][subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;		//dinuc on nuc
						hessian[revDinucIdx][subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;		//dinuc on nuc
					}
					for (int csc=0; csc<nShapeClasses; csc++) {											//shape on nuc
						hessian[shapePosIdx+csc][subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					if (subLoc < k-1) {
						hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//nuc on dinuc
						hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//nuc on dinuc
						if (loc < k-1) {
							hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//dinuc on dinuc
							hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//dinuc on dinuc
						}
						for (int csc=0; csc<nShapeClasses; csc++) {		//shape on dinuc
							hessian[shapePosIdx+csc][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum*shapeFeatures[rShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					if (loc < k-1) {
						for (int csc=0; csc<nShapeClasses; csc++) {		//dinuc on shape
							hessian[fwdDinucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
							hessian[revDinucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}				
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
		
	private double swNucleotideDinucleotideShapeNoFlank(long input) {
		double totalSum		= 0;
		double fSubSum		= 0;
		double rSubSum		= 0;
		int fShapeIdx		= 0;
		int rShapeIdx		= 0;
		long fSubString;
		long rSubString;
		long forwardStrand 	= input;
		long reverseStrand 	= reverseComplement(input, l);
		long fShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long rShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long fShapeSubString;
		long rShapeSubString;
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			totalSum += Math.exp(fSubSum) + Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		
		return totalSum+nsBindingValue;
	}
	
	private double swGradNucleotideDinucleotideShapeNoFlank(long input, int ki, double[] nucGrads, double[] dinucGrads, double[] shapeGrads) {
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		long fSubString;
		long rSubString;
		long fShapeSubString;
		long rShapeSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate string sum
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*fPartialSums[j]/totalSum;
			rSubSum = ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				nucGrads[loc*4+((int) (fSubString&3))] += fSubSum;
				nucGrads[loc*4+((int) (rSubString&3))] += rSubSum;
				dinucGrads[loc*16+((int) (fSubString&15))] += fSubSum;
				dinucGrads[loc*16+((int) (rSubString&15))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					shapeGrads[loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					shapeGrads[loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			nucGrads[(k-1)*4+((int) (fSubString&3))] += fSubSum;
			nucGrads[(k-1)*4+((int) (rSubString&3))] += rSubSum;
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				shapeGrads[(k-1)*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
				shapeGrads[(k-1)*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		return totalSum;
	}
	
	private double swHessianNucleotideDinucleotideShapeNoFlank(long input, int ki, double[] nucGrads, double[] dinucGrads, double[] shapeGrads, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx;
		int fShapeIdx			= 0;
		int rShapeIdx			= 0;
		double totalSum			= 0;
		double fSubSum			= 0;
		double rSubSum			= 0;
		double[] fPartialSums	= new double[maxFrames];
		double[] rPartialSums	= new double[maxFrames];
		double[] tempGrad		= new double[nonNSFeatures];
		long fSubString;
		long fwdBinSubString;
		long rSubString;
		long revBinSubString;
		long fShapeSubString;
		long fwdShapeBinSubString;
		long rShapeSubString;
		long revShapeBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate String Sum		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			fPartialSums[j] = Math.exp(fSubSum);
			rPartialSums[j] = Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		totalSum		= Array.sum(fPartialSums) + Array.sum(rPartialSums);
		totalSum		= totalSum+nsBindingValue;
		forwardStrand	= fwdStrand;
		reverseStrand	= revStrand;
		fShapeSeq		= fwdShapeSeq;
		rShapeSeq		= revShapeSeq;
		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*fPartialSums[j]/totalSum;
			rSubSum			= ki*rPartialSums[j]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString			= forwardStrand & frameMask;
				revBinSubString			= reverseStrand & frameMask;
				fwdShapeBinSubString	= fShapeSeq & shapeMask;
				revShapeBinSubString	= rShapeSeq & shapeMask;
				fwdNucIdx				= loc*4+((int) (fSubString&3));
				revNucIdx				= loc*4+((int) (rSubString&3));
				fwdDinucIdx				= loc*16+((int) (fSubString&15));
				revDinucIdx				= loc*16+((int) (rSubString&15));
				fShapeIdx				= (int) (fShapeSubString & 1023);
				rShapeIdx				= (int) (rShapeSubString & 1023);				
				shapePosIdx				= loc*nShapeClasses+dinucOffset;
				nucGrads[fwdNucIdx]		+= fSubSum;
				nucGrads[revNucIdx]		+= rSubSum;
				tempGrad[fwdNucIdx]		+= fSubSum;
				tempGrad[revNucIdx]		+= rSubSum;
				if (loc < k-1) {
					dinucGrads[fwdDinucIdx] += fSubSum;
					dinucGrads[revDinucIdx]	+= rSubSum;
					fwdDinucIdx				+= nucOffset;
					revDinucIdx				+= nucOffset;
					tempGrad[fwdDinucIdx]	+= fSubSum;
					tempGrad[revDinucIdx]	+= rSubSum;
				}
				for (int csc=0; csc<nShapeClasses; csc++) {
					shapeGrads[shapePosIdx+csc-dinucOffset]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= fSubSum*shapeFeatures[fShapeIdx][csc];
					shapeGrads[shapePosIdx+csc-dinucOffset]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					tempGrad[shapePosIdx+csc]				+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+dinucOffset;
					hessian[fwdNucIdx][subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;		//nuc on nuc
					if (loc < k-1) {
						hessian[fwdDinucIdx][subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;		//dinuc on nuc
						hessian[revDinucIdx][subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;		//dinuc on nuc
					}
					for (int csc=0; csc<nShapeClasses; csc++) {											//shape on nuc
						hessian[shapePosIdx+csc][subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					if (subLoc < k-1) {
						hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//nuc on dinuc
						hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//nuc on dinuc
						if (loc < k-1) {
							hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//dinuc on dinuc
							hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//dinuc on dinuc
						}
						for (int csc=0; csc<nShapeClasses; csc++) {		//shape on dinuc
							hessian[shapePosIdx+csc][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum*shapeFeatures[rShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					if (loc < k-1) {
						for (int csc=0; csc<nShapeClasses; csc++) {		//dinuc on shape
							hessian[fwdDinucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
							hessian[revDinucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}				
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
		//Calculate hessian term 2
		for (int i=0; i<nonNSFeatures; i++) {
			if (tempGrad[i]==0) continue;
			for (int j=0; j<nonNSFeatures; j++) {
				if (tempGrad[j]==0) continue;
				hessian[i][j] -= tempGrad[i]*tempGrad[j]/ki;
			}
			hessian[i][nonNSFeatures] -= tempGrad[i]/totalSum;
			hessian[nonNSFeatures][i] -= tempGrad[i]/totalSum;
		}
		return totalSum;
	}
	
	private void threadSchedule(int nThreads) {
		int nDataPoints		= R1Probes.length;
		threadRange			= new int[nThreads][2];
		int divLength 		= (int) Math.floor( ((double)nDataPoints)/((double) nThreads) );
		int remainder 		= nDataPoints % nThreads;
		
		threadRange[0][0] = 0;
		for (int i=0; i<nThreads-1; i++) {
			if (remainder>0) {
				threadRange[i][1] = threadRange[i][0] + divLength+1;
				remainder--;
			} else {
				threadRange[i][1] = threadRange[i][0] + divLength;
			}
			threadRange[i+1][0] = threadRange[i][1];
		}
		threadRange[nThreads-1][1] = nDataPoints;
	}
		
	protected void reverseBetas() {
		nucBetas 		= Array.blockReverse(nucBetas, 4);
		if (isDinuc) 	dinucBetas 	= Array.blockReverse(dinucBetas, 16);	
		if (isShape)	shapeBetas 	= Array.blockReverse(shapeBetas, nShapeClasses);
	}
	
	private void reverseGradients() {
		nucGradients	= Array.blockReverse(nucGradients, 4);
		if (isDinuc)	dinucGradients	= Array.blockReverse(dinucGradients, 16);
		if (isShape)	shapeGradients	= Array.blockReverse(shapeGradients, nShapeClasses);
	}
	
	private void reverseHessian() {
		double[][] output	= new double[totFeatures][totFeatures];
		
		for (int i=0; i<totFeatures; i++) {
			for (int j=0; j<totFeatures; j++) {
				output[i][j] = hessian[(int) revHessianIdxMap[i]][(int) revHessianIdxMap[j]];
			}
		}
		hessian = output;
	}
	
	public void normalForm() {
		int posOffset, currNV;
		long currDi;
		double[] currPos;
		double[][] nullDirs;
		QRDecomposition decomp;
		
		//First construct the null directions
		if (isDinuc) {
			nullDirs = new double[totFeatures][k+(k-1)*7];
		} else {
			nullDirs = new double[totFeatures][k];
		}
		//Construct the nucleotide directions
		for (int offset=0; offset<k; offset++) {
			for (long cn=0; cn<4; cn++) {
				nullDirs[offset*4+(int) cn][offset] = 1;
			}
			if (isNSBinding)	nullDirs[totFeatures-1][offset] = 1;
		}
		//Construct the dinucleotide directions, if needed
		if (isDinuc) {
			posOffset = k*4;
			currNV = k;
			for (int offset=0; offset<k-1; offset++) {
				//First loop over the left position
				for (long cn=0; cn<4; cn++) {
					nullDirs[offset*4+(int) cn][currNV] = -1;
					for (long vn=0; vn<4; vn++) {
						currDi = (cn<<2) | vn;
						nullDirs[(int) (posOffset+offset*16+currDi)][currNV] = 1;	
					}
					currNV++;
				}
				//Next loop over the right position
				for (long cn=0; cn<3; cn++) {
					nullDirs[(offset+1)*4+(int) cn][currNV] = -1;
					for (long vn=0; vn<4; vn++) {
						currDi = (vn<<2) | cn;
						nullDirs[(int) (posOffset+offset*16+currDi)][currNV] = 1;
					}
					currNV++;
				}
			}
		}
		//Perform QR Decomposition
		decomp = new QRDecomposition(new Matrix(nullDirs));
		nullDirs = Array.transpose(decomp.getQ().getArray());
		
		//Project out the null vector directions 
		currPos = Array.clone(getPositionVector());
		for (int i=0; i<nullDirs.length; i++) {
			currPos = Array.addScalarMultiply(currPos, -1.0*Array.dotProduct(currPos, nullDirs[i]), nullDirs[i]);
		}
		setParams(currPos);
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
	
	public class ThreadedFunctionEvaluator implements Callable<Double>{
		private int startIdx;
		private int endIdx;
		
		public ThreadedFunctionEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public Double call() throws Exception {
			double output = 0;
			
			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotide(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotide(R1Probes[i]));
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideDinucleotide(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideDinucleotide(R1Probes[i]));
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideShape(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideShape(R1Probes[i]));
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideDinucleotideShape(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideDinucleotideShape(R1Probes[i]));
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideNoFlank(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideNoFlank(R1Probes[i]));
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideDinucleotideNoFlank(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideDinucleotideNoFlank(R1Probes[i]));
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideShapeNoFlank(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideShapeNoFlank(R1Probes[i]));
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
//					output += R1R0Prob.get(i)*swNucleotideDinucleotideShapeNoFlank(R1Probes.get(i));
					output += R1Counts[i]*Math.log(R1R0Prob[i]*swNucleotideDinucleotideShapeNoFlank(R1Probes[i]));
				}
			}
			return output;
		}
	}
	
	public class ThreadedGradientEvaluator implements Callable<CompactGradientOutput>{
		private int startIdx;
		private int endIdx;
		private double probeSum;
		private double nsGradient;
		private double[] nucGradients	= null;	
		private double[] dinucGradients	= null;
		private double[] shapeGradients	= null;
		
		public ThreadedGradientEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public CompactGradientOutput call() throws Exception {
			double functionValue	= 0;
			nsGradient				= 0;
			nucGradients			= new double[4*k];
			if (isDinuc)	dinucGradients	= new double[16*(k-1)];
			if (isShape)	shapeGradients	= new double[nShapeClasses*k];
			
			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotide(R1Probes[i], R1Counts[i], nucGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideDinucleotide(R1Probes[i], R1Counts[i], nucGradients, dinucGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideShape(R1Probes[i], R1Counts[i], nucGradients, shapeGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideDinucleotideShape(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, shapeGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideNoFlank(R1Probes[i], R1Counts[i], nucGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideDinucleotideNoFlank(R1Probes[i], R1Counts[i], nucGradients, dinucGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideShapeNoFlank(R1Probes[i], R1Counts[i], nucGradients, shapeGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swGradNucleotideDinucleotideShapeNoFlank(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, shapeGradients);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			}
			return (new CompactGradientOutput(functionValue, toVector(isNSBinding, nsGradient, nucGradients, dinucGradients, shapeGradients)));
		}
	}
	
	public class ThreadedHessianEvaluator implements Callable<CompactGradientOutput>{
		private int startIdx;
		private int endIdx;
		private double probeSum;
		private double nsGradient;
		private double[] nucGradients	= null;	
		private double[] dinucGradients	= null;
		private double[] shapeGradients	= null;
		private double[][] hessian		= null;
		
		public ThreadedHessianEvaluator(int startIdx, int endIdx) {
			this.startIdx				= startIdx;
			this.endIdx					= endIdx;
		}
		
		@Override
		public CompactGradientOutput call() throws Exception {
			double functionValue	= 0;
			nsGradient				= 0;
			nucGradients			= new double[4*k];
			if (isDinuc)	dinucGradients	= new double[16*(k-1)];
			if (isShape)	shapeGradients	= new double[nShapeClasses*k];
			hessian					= new double[nonNSFeatures+1][nonNSFeatures+1];

			if (type==0) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotide(R1Probes[i], R1Counts[i], nucGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==1) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideDinucleotide(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==2) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideShape(R1Probes[i], R1Counts[i], nucGradients, shapeGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==3){
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideDinucleotideShape(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, shapeGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==4){
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideNoFlank(R1Probes[i], R1Counts[i], nucGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==5) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideDinucleotideNoFlank(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==6) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideShapeNoFlank(R1Probes[i], R1Counts[i], nucGradients, shapeGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			} else if(type==7) {
				for (int i=startIdx; i<endIdx; i++) {
					probeSum		= swHessianNucleotideDinucleotideShapeNoFlank(R1Probes[i], R1Counts[i], nucGradients, dinucGradients, shapeGradients, hessian);
					nsGradient		+= R1Counts[i]*(1/probeSum-1/Z_FFM);
					hessian[nonNSFeatures][nonNSFeatures] += R1Counts[i]*(1/Z_FFM - 1/probeSum + nsBindingValue*(1/(probeSum*probeSum) - 1/(Z_FFM*Z_FFM)));
					functionValue	+= R1Counts[i]*Math.log(R1R0Prob[i]*probeSum);
				}
			}
			return (new CompactGradientOutput(functionValue, toVector(isNSBinding, nsGradient, nucGradients, dinucGradients, shapeGradients), hessian));
		}
	}	
}