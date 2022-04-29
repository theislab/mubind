package base;

import java.util.Random;

public class Array {
	public static double[] abs(double[] input) {
		double[] output = new double[input.length];
		for (int i=0; i<input.length; i++) {
			output[i] = Math.abs(input[i]);
		}
		return output;
	}
	
	public static double[] add(double[] a, double[] b) {
		if (a.length != b.length){
			throw new IllegalArgumentException("Array lengths do not match.");
		}
		double[] output = new double[a.length];
		
		for (int i=0; i<a.length; i++) {
			output[i] = a[i]+b[i];
		}
		return output;
	}
	
	/**
	 * addScalarMultiply takes an input vectors x and y and a constant c to
	 * return the value
	 * 		x + c*y
	 * 
	 * @param x the original vector
	 * @param c	the scalar which will multiply the vector y
	 * @param y	the vector which will be multiplied by c and added to x
	 * @return a double array containing the value of x + c*y
	 */

	public static double[] addScalarMultiply(double[] x, double c, double[] y) {
		double[] output = new double[x.length];
		
		for (int i=0; i<x.length; i++) {
			output[i] = x[i]+c*y[i];
		}
		return output;
	}
	
	public static double[] multiply(double[] x, double[] y) {
		double[] output = new double[x.length];
		
		for (int i=0; i<x.length; i++) {
			output[i] = x[i]*y[i];
		}
		return output;
	}
	
	//Flip orientation of array by block
	public static double[] blockReverse(double[] input, int blockSize) throws IllegalArgumentException {
		double[] output = new double[input.length];
		if (input.length % blockSize != 0){
			throw new IllegalArgumentException("Input array is not a multiple of block size.");
		}
		int divs = input.length/blockSize-1;
		
		//flip orientation
		for (int offset=0; offset<=divs; offset++) {
			for (int idx=0; idx<blockSize; idx++) {
				output[offset*blockSize+idx] = input[(divs-offset)*blockSize+idx];
			}
		}
		return output;
	}
	
	//Flip orientation of array by block
	public static int[] blockReverse(int[] input, int blockSize) throws IllegalArgumentException {
		int[] output = new int[input.length];
		if (input.length % blockSize != 0){
			throw new IllegalArgumentException("Input array is not a multiple of block size.");
		}
		int divs = input.length/blockSize-1;
		
		//flip orientation
		for (int offset=0; offset<=divs; offset++) {
			for (int idx=0; idx<blockSize; idx++) {
				output[offset*blockSize+idx] = input[(divs-offset)*blockSize+idx];
			}
		}
		return output;
	}
	
	public static double[] cat(double[] a, double[] b) {
		int lenA = (a==null) ? 0 : a.length;
		int lenB = (b==null) ? 0 : b.length;
		double[] output = new double[lenA+lenB];
		
		for (int i=0; i<lenA; i++) {
			output[i] = a[i];
		}
		for (int j=0; j<lenB; j++) {
			output[lenA+j] = b[j];
		}
		return output;
	}
	
	public static double[] cat(double[] a, double b) {
		int lenA = (a==null) ? 0 : a.length;
		double[] output = new double[lenA+1];
		
		for (int i=0; i<lenA; i++) {
			output[i] = a[i];
		}
		output[lenA] = b;
		return output;
	}
	
	public static int[] cat(int[] a, int[] b) {
		int lenA = (a==null) ? 0 : a.length;
		int lenB = (b==null) ? 0 : b.length;
		int[] output = new int[lenA+lenB];
		
		for (int i=0; i<lenA; i++) {
			output[i] = a[i];
		}
		for (int j=0; j<lenB; j++) {
			output[lenA+j] = b[j];
		}
		return output;
	}
	
	public static int[] cat(int[] a, int b) {
		int lenA = (a==null) ? 0 : a.length;
		int[] output = new int[lenA+1];
		
		for (int i=0; i<lenA; i++) {
			output[i] = a[i];
		}
		output[lenA] = b;
		return output;
	}
	
	public static String[] cat(String[] a, String b) {
		int lenA = (a==null) ? 0 : a.length;
		String[] output = new String[lenA+1];
		
		for (int i=0; i<lenA; i++) {
			output[i] = a[i];
		}
		output[lenA] = b;
		return output;
	}
	
	//Array cloning routine **DEEP COPY ROUTINE
	public static double[] clone(double[] js) {
		double[] output = new double[js.length];
		
		for (int i = 0; i < js.length; i++){
			output[i] = js[i];
		}
		return output;
	}
	
	public static int[] clone(int[] js) {
		int[] output = new int[js.length];
		
		for (int i = 0; i < js.length; i++){
			output[i] = js[i];
		}
		return output;
	}
	
	public static String[] clone(String[] js) {
		if (js==null) {
			return null;
		}
		String[] output = new String[js.length];
		
		for (int i = 0; i < js.length; i++){
			output[i] = js[i];
		}
		return output;
	}

	public static double[][] clone(double[][] input) {
		int x = input.length, y = input[0].length;
		double[][] output = new double[x][y];
		
		for (int i=0; i<x; i++) {
			for (int j=0; j<y; j++) {
				output[i][j] = input[i][j];
			}
		}
		return output;
	}
	
	public static int[][][] clone(int[][][] input) {
		int x = input.length, y = input[0].length, z = input[0][0].length;
		int[][][] output = new int[x][y][z];
		
		for (int i=0; i<x; i++) {
			for (int j=0; j<y; j++) {
				for (int k=0; k<z; k++) {
					output[i][j] = input[i][j];
				}
			}
		}
		return output;
	}
	
	public static boolean containsTrue(boolean[] input) {
		for (boolean currBool : input) {
			if (currBool) {
				return true;
			}
		}
		return false;
	}

	// 0 1 2 3 4 5 6
	public static double[] cycleRight(double[] input, int units) {
		double[] output = new double[input.length];
		
		for (int i=input.length-1; i>=units; i--) {
			output[i] = input[i-units];
		}
		return output;
	}
	
	public static double[] cycleLeft(double[] input, int units) {
		double[] output =  new double[input.length];
		
		for (int i=0; i<input.length-units; i++) {
			output[i] = input[i+units];
		}
		return output;
	}
	
	//Calculate Euclidean Distance
	public static double dist(double[] a, double[] b) {
		double output = 0;
		for (int i = 0; i < a.length; i++) {
			output += (a[i] - b[i])*(a[i] - b[i]);
		}
		return output;
	}
	
	public static double[] divide(double[] a, double[] b) {
		double[] output = new double[a.length];
		for (int i=0; i<a.length; i++) {
			output[i] = a[i]/b[i];
		}
		return output;
	}
	
	public static double dotProduct(double[]a, double[]b) {
		double output = 0;
		for (int i=0; i<a.length; i++) {
			output += a[i]*b[i];
		}
		return output;
	}
	
	//Exponentiate every element in the array
	public static double[] exp(double[] input) {
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			output[i] = Math.exp(input[i]);
		}
		return output;
	}
	
	//Compute the frobenius norm of a matrix
	public static double frobenius(double[][] input) {
		double output = 0;
		
		for (int i=0; i<input.length; i++) {
			for (int j=0; j<input[0].length; j++) {
				output += input[i][j]*input[i][j];
			}
		}
		return Math.sqrt(output);
	}
	
	public static double[] grow(double[] input, int units) {
		double[] output = new double[input.length+units*2];
		
		for (int i=0; i<input.length; i++) {
			output[i+units] = input[i];
		}
		return output;
	}
	
	public static double length(double[] input) {
		double output = 0;
		for (int i=0; i<input.length; i++) {
			output += input[i]*input[i];
		}
		return output;
	}
	
	//Find the natural log of every element in the array
	public static double[] log(double[] input) {
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++) {
			output[i] = Math.log(input[i]);
		}
		return output;
	}

	//Return the maximum value of the array
	public static double max(double[] input) {
		double output = input[0];
		for (int i=1; i<input.length; i++) {
			if (input[i] > output) {
				output = input[i];
			}
		}
		return output;
	}
	
	//Return the maximum value of the array
	public static int max(int[] input) {
		int output = input[0];
		for (int i=1; i<input.length; i++) {
			if (input[i] > output) {
				output = input[i];
			}
		}
		return output;
	}
	
	//Max-Normalize the array
	public static double[] maxNormalize(double[] input) {
		double[] output = new double[input.length];
		double maxVal = input[0];
		
		for (int i = 1; i < input.length; i++) {
			if (input[i] > maxVal) {
				maxVal = input[i];
			}
		}
		
		for (int i = 0; i < input.length; i++) {
			output[i] = input[i]/maxVal;
		}
		return output;
	}
	
	public static double mean(double[] input) {
		double output = 0;
		for (int i=0; i<input.length; i++) {
			if (Double.isNaN(input[i])) {
				continue;
			} else {
				output += input[i];
			}
		}
		return (output/input.length);
	}
	
	//Return the minimum value of the array
	public static double min(double[] input) {
		double output = input[0];
		for (int i=1; i<input.length; i++) {
			if (input[i] < output) {
				output = input[i];
			}
		}
		return output;
	}
	
	public static double[] normalize(double[] input) {
		double sum		= 0;
		double[] output = new double[input.length];
		
		for (int i=0; i<input.length; i++) {
			sum += input[i]*input[i];
		}
		sum = Math.sqrt(sum);
		for (int i=0; i<input.length; i++) {
			output[i] = input[i]/sum;
		}
		return output;
	}
	
	public static double norm(double[] input) {
		double output = 0;
		
		for (int i=0; i<input.length; i++) {
			output += input[i]*input[i];
		}
		return Math.sqrt(output);
	}
	
	public static double[] ones(int nDim) {
		double[] output = new double[nDim];
		for (int i=0; i<nDim; i++) {
			output[i] = 1;
		}
		return output;
	}

	//Print all elements in the array
	public static void print(boolean[] input) {
		for (int i=0; i<input.length; i++) {
			System.out.print(input[i]+"\t");
		}
		System.out.print("\n");
	}
	
	public static void print(int[] input) {
		for (int i=0; i<input.length; i++) {
			System.out.print(input[i]+"\t");
		}
		System.out.print("\n");
	}
	
	public static void print(int[][] input) {
		for (int i=0; i<input.length; i++) {
			print(input[i]);
		}
	}
	
	public static void print(double[] input) {
		for (int i=0; i<input.length; i++) {
			System.out.print(input[i]+"\t");
		}
		System.out.print("\n");
	}
	
	public static void print(double[][] input) {
		for (int i=0; i<input.length; i++) {
			print(input[i]);
		}
	}
	
	public static void print(long[] input) {
		for (int i=0; i<input.length; i++) {
			System.out.print(input[i]+"\t");
		}
		System.out.print("\n");
	}
	
	public static void print(long[][] input) {
		for (int i=0; i<input.length; i++) {
			print(input[i]);
		}
	}

	public static void print(String[] input) {
		for (int i=0; i<input.length; i++) {
			System.out.print(input[i]+"\t");
		}
		System.out.print("\n");
	}
	
	//Return an array of length l with random doubles
	public static double[] randomDouble(int l) {
		double[] output		= new double[l];
		Random generator	= new Random();
		
		for (int i=0; i<l; i++) {
			output[i] = generator.nextDouble();
		}
		return output;
	}
	
	//Reverse order of array in place
	public static double[] reverse(double[] in) {
		int rIdx = in.length-1;
		int lIdx = 0;
		double temp;
		
		while (lIdx<rIdx) {
			temp = in[lIdx];
			in[lIdx] = in[rIdx];
			in[rIdx] = temp;
			lIdx++;
			rIdx--;
		}
		
		return in;
	}

	public static double[] scalarMultiply(double[] input, double scalar) {
		double[] output = new double[input.length];
		for (int i=0; i<input.length; i++) {
			output[i] = input[i]*scalar;
		}
		return output;
	}
	
	public static double[][] scalarMultiply(double[][] input, double scalar) {
		double[][] output = new double[input.length][input[0].length];
		
		for (int i=0; i<input.length; i++) {
			for (int j=0; j<input[0].length; j++) {
				output[i][j] = input[i][j]*scalar;
			}
		}
		return output;
	}
	
	public static double sum(double[] input) {
		double output = 0;
		for (int i=0; i<input.length; i++) {
			output += input[i];
		}
		return output;
	}
	
	public static double sum(int[] input) {
		int output = 0;
		for (int i=0; i<input.length; i++) {
			output += input[i];
		}
		return output;
	}
	
	public static double[] subtract(double[] a, double[] b) {
		if (a.length != b.length){
			throw new IllegalArgumentException("Array lengths do not match.");
		}
		double[] output = new double[a.length];
		
		for (int i=0; i<a.length; i++) {
			output[i] = a[i]-b[i];
		}
		return output;
	}
	
	public static double[][] subtract(double[][] a, double[][] b) {
		double[][] output = new double[a.length][a[0].length];
		
		for (int i=0; i<a.length; i++) {
			for (int j=0; j<a[0].length; j++) {
				output[i][j] = a[i][j]-b[i][j];
			}
		}
		return output;
	}
	
	//Symmetrize a matrix via averaging
	public static double[][] symmetrize(double[][] input) {
		int size = input.length;
		if (size!=input[0].length) {
			throw new IllegalArgumentException("Input matrix is not square!");
		}
		double[][] output = new double[size][size];
		
		for (int i=0; i<size; i++) {
			for (int j=0; j<size; j++) {
				output[i][j] = (input[i][j]+input[j][i])/2;
				output[j][i] = output[i][j];
			}
		}
		return output;
	}

	public static double[][] transpose(double[][] input) {
		double[][] output = new double[input[0].length][input.length];
		
		for (int x=0; x<input.length; x++) {
			for (int y=0; y<input[0].length; y++) {
				output[y][x] = input[x][y];
			}
		}
		return output;
	}
	
	//Normalize by a value
	public static double[] valueNormalize(double[] input, double value) {
		double[] output = new double[input.length];
		
		for (int i=0; i<input.length; i++) {
			output[i] = input[i] - value;
		}
		return output;
	}
	
	//What is the largest element?
	public static int whichMax(double[] input) {
		double maxVal	= input[0];
		int index		= 0;
		
		for (int i=1; i<input.length; i++) {
			if (input[i] > maxVal) {
				maxVal	= input[i];
				index	= i;
			}
		}
		return index;
	}
	
	//What is the smallest element?
	public static int whichMin(double[] input) {
		double minVal = input[0];
		int index	  = 0;
		
		for (int i=1; i<input.length; i++) {
			if (input[i] < minVal) {
				minVal = input[i];
				index = i;
			}
		}
		return index;
	}
}
