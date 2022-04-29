package minimizers;

import java.util.ArrayList;
import java.util.Formatter;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import base.*;

public class Minimizer {
	protected boolean crossValidate = false;
	protected double epsilon, randomSeedScale = .05;
	protected Model model;
	
	/* This is the function where the actual minimizer algorithm must be
	 * defined. The public functions below are wrapper functions that
	 * automatically provide the user with added functionality (see below).
	 */
	protected Fit doMinimize(double[] seed, String trajectoryFile) 
			throws Exception {
		return null;
	}
	
	//Special case of minimize to remove negative eigenvalues
	public Fit[] minimize(double[] seed, String trajectoryFile) 
			throws Exception {
		boolean notConverged= false;
		int nIterations = 0, retryCount = 0;
		double originalSeedScale = randomSeedScale;
		int[] nFunctionEvals= new int[1];
		double[] realEV, currLoc, badDirection;
		double[][] hessian, VMatrix;
		Fit fitOutput;
		EigenvalueDecomposition ev; 
		
		/* Try reseeding the fit if no seed is provided, as sometimes the random
		 * start position can make it hard for LBFGS to converge. This will not 
		 * trigger for Pattern Search. This refitting step will be attempted 
		 * maxRetries number of times. Additionally, the random seed magnitude
		 * will drop by a factor of 10 every failure. 
		 */
		if (seed==null) {
			while (true) {
				try {
					fitOutput = doMinimize(seed, trajectoryFile);
					break;
				} catch (ArithmeticException e) {
					if (retryCount>model.maxFitRetries) {
						throw new Exception("Random reseed failed: General "
								+ "Minimizer Failure.", e);
					}
					System.out.println("Minimizer failed. Attempting refit with"
							+ " new random seed.");
					retryCount++;
					randomSeedScale /= 10;
					continue;
				} catch (Exception e) {
					throw e;
				}
			}
			randomSeedScale = originalSeedScale;
		} else {
			fitOutput = doMinimize(seed, trajectoryFile);
		}
		
		/* The negative eigenvalue analysis is only performed on non-cross 
		 * validation runs. Additionally, if a hessian is undefined (i.e. it
		 * returns null), this step is skipped.
		 */
		if (!crossValidate) {
			if (model.getHessian()==null) {

				return new Fit[]{fitOutput};
			}
			
			//All work is in REDUCED space. Symmetrize to reduce errors
			hessian = model.getHessian().hessian;
			hessian = model.compressHessian(hessian);
			hessian = Array.symmetrize(hessian);
			//reduced space
			ev		= new EigenvalueDecomposition(new Matrix(hessian));
			
			realEV	= Array.maxNormalize(ev.getRealEigenvalues());
			VMatrix	= ev.getV().getArray();
			currLoc	= model.compressPositionVector(model.getPositionVector());
			//Does a negative eigenvalue exist?
			for (int w=0; w<realEV.length; w++) {
				if (realEV[w] < -model.evCutoff) {
					//A negative eigenvalue exists, perform linesearch in -EV eigenvector direction
					notConverged	= true;
					badDirection	= new double[realEV.length];
					for (int x=0; x<realEV.length; x++) {
						badDirection[x] = VMatrix[x][w];
					}
					currLoc = negativeEVLineSearch(currLoc, badDirection, nFunctionEvals);
				}
			}
			if (notConverged) {
				System.out.println("Model has not converged. Restarting minimizer.");
				currLoc 			= model.uncompress(currLoc);		//Return to full space
				epsilon				/= 2;
				nIterations			+= fitOutput.fitSteps;
				nFunctionEvals[0]	+= fitOutput.functionCalls;
				fitOutput			= minimize(currLoc, trajectoryFile)[0];
				epsilon				*= 2;
				nIterations			+= fitOutput.fitSteps;
				nFunctionEvals[0]	+= fitOutput.functionCalls;
				fitOutput.fitSteps		= nIterations;
				fitOutput.functionCalls	= nFunctionEvals[0];
			}
		}
		return new Fit[]{fitOutput};
	}
	
	public Fit[] minimize(double[] seed, String trajectoryFile, ArrayList<Object[]> datasets) throws Exception {
		double testL = 0, adjTestL = 0, trainOutput;
		Fit output = null;
		
		System.out.println("Fitting full dataset.");
		output = minimize(seed, trajectoryFile)[0];
		
		//loop over CV datasets, if they exist, and report average values
		if (datasets.size()>1) {
			crossValidate = !crossValidate;
			for (int i=1; i<datasets.size(); i++) {
				System.out.println("Fitting on CV dataset "+i);
				//Fit on new CV training dataset
				model.replaceData(datasets.get(i)[0]);
				minimize(output.positionVector(), null);
				//Evaluate on test dataset
				model.replaceData(datasets.get(i)[1]);
				trainOutput	= model.functionEval();
				testL		+= trainOutput*model.likelihoodNormalizer();
				adjTestL	+= (trainOutput-model.maxLikelihood())*model.likelihoodNormalizer();
			}
			output.recordCrossValidate(testL/(datasets.size()-1), adjTestL/(datasets.size()-1));
			model.replaceData(datasets.get(0)[0]);
			crossValidate = !crossValidate;
		}
		return new Fit[]{output};
	}
	
	public Fit[] shiftPermutation(double[] seed, String trajectoryFile, int nCycles, ArrayList<Object[]> datasets,
			Results results) throws Exception {
		String trajectoryPath;
		Fit baseFit;
		Fit[] outputs = new Fit[2*nCycles+1];
		
		//Fit the 0'th binding frame
		trajectoryPath = (trajectoryFile==null) ? null : trajectoryFile+"0";
		outputs[nCycles] = minimize(seed, trajectoryPath, datasets)[0];
		outputs[nCycles].recordShiftIndex(0);
		if (results!=null)	results.addFit(outputs[nCycles]);
		
		//Fit all other shifts
		for (int cycleNumber=1; cycleNumber<=nCycles; cycleNumber++) {
			//Fit the left shift
			if (outputs[nCycles-cycleNumber+1]!=null) {
				baseFit		= outputs[nCycles-cycleNumber+1];
				trajectoryPath = (trajectoryFile==null) ? null : trajectoryFile+"-"+cycleNumber;
				try {
					outputs[nCycles-cycleNumber] = minimize(model.shiftBetas(baseFit.positionVector(), -1), trajectoryPath, datasets)[0];
					outputs[nCycles-cycleNumber].recordShiftIndex(-cycleNumber);
					if (results!=null)	results.addFit(outputs[nCycles-cycleNumber]);				
				} catch (Exception e) {
					e.printStackTrace();
					outputs[nCycles-cycleNumber] = null;
				}
			}
			
			//Fit the right shift
			if (outputs[nCycles+cycleNumber-1]!=null) {
				baseFit		= outputs[nCycles+cycleNumber-1];
				trajectoryPath = (trajectoryFile==null) ? null : trajectoryFile+"+"+cycleNumber;
				try {
					outputs[nCycles+cycleNumber] = minimize(model.shiftBetas(baseFit.positionVector(), 1), trajectoryPath, datasets)[0];
					outputs[nCycles+cycleNumber].recordShiftIndex(cycleNumber);
					if (results!=null)	results.addFit(outputs[nCycles+cycleNumber]);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		return outputs;
	}
		
	protected double[] negativeEVLineSearch(double[] start, double[] badDir, int[] iterationCounter) throws Exception {
		boolean isCubic = true;
		int iterations = 0, maxIters = 200, cubicIterations;
		double alphaMin = 0, alphaMax = 0, alphaTest, fMin = 0, fMax = 0, fTest, gradMin = 0, gradMax = 0, gradTest, eta, nu, s;
		double[] searchDir = Array.clone(badDir);
		Model.CompactGradientOutput out;
		
		/*Calculate the gradient at alpha0 to determine the minimization 
		 * direction. This is either positive or negative (reverse) direction.*/
		out		= gradientEval(start, false);
		fMin	= out.functionValue;
		gradMin	= Array.dotProduct(out.gradientVector, searchDir);
		if (gradMin>0) {									//Flip search direction if gradient is positive
			searchDir		= Array.scalarMultiply(searchDir, -1);
			gradMin			= Array.dotProduct(out.gradientVector, searchDir);
		}
		alphaTest = 1E-1;
		//Select initial bracket
		while (Math.abs(alphaTest)<1E6) {
			out		= gradientEval(Array.addScalarMultiply(start, alphaTest, searchDir), false);
			fTest	= out.functionValue;
			gradTest= Array.dotProduct(out.gradientVector, searchDir);
			//Check to see if the gradient has flipped signs.
			if (Math.signum(gradMin*gradTest) < 0) {		
				if (gradTest < 0) {
					alphaMax= 0;
					fMax	= fMin;
					gradMax	= gradMin;
					alphaMin= alphaTest;
					fMin	= fTest;
					gradMin	= gradTest;
				} else {
					alphaMin= 0;
					alphaMax= alphaTest;
					fMax	= fTest;
					gradMax	= gradTest;
				}
				break;
			}
			alphaTest *= 10;
		}
//		System.out.println("Initial bracket selected. Min: "+alphaMin+", Max: "+alphaMax);
		//Bisection + Cubic interpolation to find minimum
		cubicIterations = 0;
		while (iterations<maxIters) {
			//Calculate new cubic-interpolation alpha point
			if (isCubic) {
				eta 		= gradMin + gradMax + 3*(fMin-fMax)/(alphaMax-alphaMin);
				s			= Math.max(Math.abs(eta), Math.max(Math.abs(gradMin), Math.abs(gradMax)));
				nu			= s*Math.sqrt((eta/s)*(eta/s)-(gradMax/s)*(gradMin/s));
				if (alphaMax<alphaMin)	nu = -nu;
				alphaTest	= alphaMin + (alphaMax-alphaMin)*((eta + nu - gradMin)/(gradMax + 2*nu - gradMin));
				out			= gradientEval(Array.addScalarMultiply(start, alphaTest, searchDir), false);
				fTest		= out.functionValue;
				gradTest	= Array.dotProduct(out.gradientVector, searchDir);
				iterations++;
				cubicIterations++;
			} else {
				alphaTest	= (alphaMax+alphaMin)/2;
				out			= gradientEval(Array.addScalarMultiply(start, alphaTest, searchDir), false);
				fTest		= out.functionValue;
				gradTest	= Array.dotProduct(out.gradientVector, searchDir);
				iterations++;
			}
			//Cubic/Bisection Control
			if (Math.abs((alphaMin-alphaTest)/alphaMin)<1E-5 || Math.abs((alphaMax-alphaTest)/alphaMax)<1E-5) {
				isCubic			= false;
				cubicIterations = 4;
			}
			if (cubicIterations==0 && !isCubic) {
				isCubic			= true;
			}
			if (cubicIterations==3 && isCubic) {
				isCubic			= false;
				cubicIterations = 0;
			}
			//Check for convergence
//			System.out.println("gradTest: "+gradTest/Array.norm(Array.add(start, Array.scalarMultiply(searchDir, alphaTest))));
			if (Math.abs(gradTest)/Array.norm(Array.addScalarMultiply(start, alphaTest, searchDir)) < epsilon ||
					Math.abs(alphaMax-alphaMin) < 1E-3) {
				/*The case where gradTest is greater than 0, so it will replace
				 * grad max. If so, choose the lower of gradTest and gradMin */
				if (gradTest>0 && Math.abs(gradTest) > Math.abs(gradMin)) {	
					alphaTest = alphaMin;
				}
				/*The case where gradTest is less than 0, so it will replace 
				 * grad min. If so, choose the lower of gradTest and gradMax */
				if (gradTest<0 && Math.abs(gradTest) > Math.abs(gradMax)) {
					alphaTest = alphaMax;
				}
//				System.out.println("Final alpha: "+alphaTest);
				iterationCounter[0] = iterations;
				return Array.addScalarMultiply(start, alphaTest, searchDir);
			}
			//Else Re-bracket
			if (gradTest > 0) {
				alphaMax= alphaTest;
				fMax	= fTest;
				gradMax	= gradTest;
			} else {
				alphaMin= alphaTest;
				fMin	= fTest;
				gradMin	= gradTest;
			}
//			System.out.println("New bracket selected. Min: "+alphaMin+", Max: "+alphaMax);
		}
		throw new Exception("Number of line search iterations exceeded!");
	}
	
	/* gradientEval uncompresses the input (desymmetrizes) and compresses
	 * the gradient (symmetrizes) so that functions work in the compressed
	 * (reduced) space.
	 */
	protected Model.CompactGradientOutput gradientEval(double[] input, 
			boolean normalize) throws Exception {
		double[] tempGradient;
		Model.CompactGradientOutput out;
		
		input				= model.uncompress(input);
		model.setParams(input);
		out 				= model.gradientEval();
		//Handle the case where the gradient has not been defined
		if (out==null)	throw new UnsupportedOperationException("The function "
				+ "gradientEval() has not been defined!");
		tempGradient		= model.compressGradient(out.gradientVector);
		if (normalize) {
			out.functionValue	*= model.likelihoodNormalizer();
			tempGradient		= Array.scalarMultiply(tempGradient, model.likelihoodNormalizer());
		}
		out.gradientVector	= tempGradient;
		return out;	
	}
	
	protected void printStep(int iterations, int calls, double likelihood, 
			double distance, double ... params) {
		Formatter fmt = new Formatter();
		
		System.out.printf("   %7d      %7d   ", iterations, calls);
		fmt.format("%18.18s   %18.18s", String.format("%10.15f", likelihood), 
				String.format("%10.15f", distance));
		System.out.print(fmt);
		fmt = new Formatter();
		for (int i=0; i<params.length; i++) {
			fmt.format("   %18.18s", String.format("%10.15f", params[i]));
		}
		System.out.print(fmt+"\n");
		fmt.close();
	}
}
