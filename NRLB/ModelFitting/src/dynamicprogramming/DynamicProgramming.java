package dynamicprogramming;

import base.Array;

public interface DynamicProgramming {
	double recursiveZ();
	void recursiveGradient();
	void recursiveHessian();
	
	double getZ();
	double[] getNucGradients();
	double[] getDinucGradients();
	double[] getShapeGradients();
	double[][] getHessian();
	
	void setAlphas(double[] nucAlphas, double[] dinucAlphas, double[] shapeBetas);
	
	default long reverse(long input, int length) {
		long output = 0;
		
		for (int i=0; i<length; i++) {
			output = ((output << 2) | (input & 3));
			input = input >> 2;
		}
		return output;
	}
	
	default long reverseComplement(long input, int length) {;
		long output = 0;
		input = ~input;
		output = output | (input & 3);
		for (int i=1; i<length; i++) {
			input = input >> 2;
			output = output << 2;
			output = output | (input & 3);
		}
		return output;
	}
	
	default double[][] matrixClone(double[][] input) {
		double[][] output = new double[input.length][input[0].length];
		
		for (int i=0; i<input.length; i++) {
			output[i] = Array.clone(input[i]);
		}
		
		return output;
	}
}
