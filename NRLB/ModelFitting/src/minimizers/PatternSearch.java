package minimizers;

import java.util.Random;

import base.Array;
import base.Fit;
import base.Model;

public class PatternSearch extends Minimizer{
	private boolean errorBars, storeHessian, isVerbose, useRandom;
	private int totFeatures;
	private double d0, theta;
	
	public PatternSearch(Model model, double d0, double dTol, double theta, boolean useRandom, 
			boolean errorBars, boolean storeHessian, boolean isVerbose) {
		this.model				= model;
		totFeatures				= model.getTotFeatures();
		this.d0					= d0;
		this.epsilon			= dTol;
		this.theta				= theta;
		this.useRandom			= useRandom;
		this.errorBars			= errorBars;
		this.storeHessian		= storeHessian;
		this.isVerbose			= isVerbose;
		crossValidate			= false;
	}
	
	protected Fit doMinimize(double[] seed, String trajectoryFile) throws Exception{
		boolean hasImproved		= false;						//Has the current step improved the estimate?
		int totalSteps			= 0;							//total number of steps taken
		int stepPos;
		int functionCalls		= 1;
		double d 				= d0;							//step size
		double delta			= 0;							//distance moved per step
		double forwardStep		= 0;
		double reverseStep		= 0;
		double tStart			= System.currentTimeMillis();
		double[] prevLoc;
		double[] currBetas		= null;
		double[] forwardBetas	= null;
		double[] reverseBetas	= null;
		double currBestFVal 	= 0;
		int[] featureOrder		= null;
		Random generator		= new Random();
		Fit fitOutput			= model.generateFit(seed);
		
		//Check to see if seed is of correct length
		if (seed!=null) {
			if (seed.length!=totFeatures) {
				System.out.println(seed.length+"\t"+totFeatures);
				throw new IllegalArgumentException("Improper seed!");
			}
			currBetas	= Array.clone(seed);
		} else {
			currBetas	= new double[totFeatures];
		}
		//Symmetrize position and evaluate function
		currBetas		= model.symmetrize(currBetas);
		model.setParams(currBetas);
		currBestFVal	= model.functionEval();
		if (isVerbose){
			System.out.println("Starting Function Value: "+currBestFVal);
			System.out.println("Iterations   Fnc. Calls           Likelihood"
					+ "       Distance Moved            Step Size");
		}
		prevLoc			= Array.clone(currBetas);
		
		while (true) {
			if (d < epsilon) {											//check to see if convergence has been reached
				tStart		= (System.currentTimeMillis()-tStart)/1000;
				model.normalForm();
				fitOutput.recordFit(totalSteps, functionCalls, tStart, currBestFVal, model);
				if (storeHessian) {
					if (errorBars) {
						model.errorEval();
						fitOutput.recordErrorBars(model.nullVectors, model.getErrorBars());
						fitOutput.storeHessian(model.getHessian());
					} else {
						model.hessianEval();
						fitOutput.storeHessian(model.getHessian());
					}
				} else if (errorBars) {
					model.errorEval();
					fitOutput.recordErrorBars(model.nullVectors, model.getErrorBars());
				}
				if (trajectoryFile!=null)	fitOutput.printTrajectories(trajectoryFile, true);
				return fitOutput;
			} else {												//if not, begin mutation steps
				hasImproved = false;								//reset	
				//Loop over all nucleotide features
				if (useRandom)	featureOrder = getRandomPermutation(totFeatures, generator);
				for (int phi=0; phi<totFeatures; phi++) {
					stepPos	= (useRandom) ? featureOrder[phi] : phi;
					//Check to see if position has already been dealt with
					if (model.isSymmetrized(stepPos)) {
						continue;
					}
					//Compute forward and reverse betas (and symmetrize)
					forwardBetas = model.orthogonalStep(currBetas, stepPos, d);
					forwardBetas = model.symmetrize(forwardBetas);
					model.setParams(forwardBetas);
					forwardStep	 = model.functionEval();
					reverseBetas = model.orthogonalStep(currBetas, stepPos, -d);
					reverseBetas = model.symmetrize(reverseBetas);
					model.setParams(reverseBetas);
					reverseStep	 = model.functionEval();
					functionCalls+= 2;
					//Forward direction improved log likelihood
					if((currBestFVal-forwardStep)/currBestFVal > 5e-16) {
						//Reverse direction is even better
						if ((forwardStep-reverseStep)/forwardStep > 5e-16) {
							currBetas	= reverseBetas;
							currBestFVal= reverseStep;
							hasImproved = true;
						} else {									//forward direction is better
							currBetas	= forwardBetas;
							currBestFVal= forwardStep;
							hasImproved	= true;
						}
					} else if ((currBestFVal-reverseStep)/currBestFVal > 5e-16) { //reverse direction improved log likelihood
						currBetas	= reverseBetas;
						currBestFVal= reverseStep;
						hasImproved	= true;
					}
				}				
				
				if (hasImproved) {									//has there been an reduction in the function value?
					totalSteps++;
					model.setParams(currBetas);
					model.normalForm();
					currBetas	= model.getPositionVector();
					delta 		= Array.dist(prevLoc, currBetas);
					prevLoc		= Array.clone(currBetas);
					if (isVerbose) {
						printStep(totalSteps, functionCalls, currBestFVal, delta, d);
					}
					fitOutput.addStep(model.getPositionVector());
					if (trajectoryFile!=null)	fitOutput.printTrajectories(trajectoryFile, true);
				} else {
					d = d*theta;									//and reduce step size
				}
			}
		}	
	}
	
	public static int[] getRandomPermutation (int length, Random generator){
	    int idx		= 0;
	    int swap	= 0;
	    int[] array = new int[length];
	 
	    // initialize array and fill it with {0,1,2...}
	    for(int i = 0; i < array.length; i++) {
	    	array[i] = i;
	    }

	    for(int i = 0; i<length; i++){
	        // randomly chosen position in array whose element
	        // will be swapped with the element in position i
	        // note that when i = 0, any position can chosen (0 thru length-1)
	        // when i = 1, only positions 1 through length -1
	        idx = i+generator.nextInt(length-i);

	        // perform swap
	        swap		= array[i];
	        array[i] 	= array[idx];
	        array[idx] 	= swap;
	    }                       
	    return array;
	}
}