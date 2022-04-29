package base;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;

public abstract class Fit implements Serializable{
	private static final long serialVersionUID	= 2591999005040132009L;
	public boolean testShifts, isCrossValidate;
	public int fitSteps, functionCalls, shiftPos, nullVectors;
	public double fitTime, trainLikelihood, testLikelihood, adjustedTestLikelihood;
	public double[] finalPosition				= null;
	public double[][] hessian					= null;
	public ArrayList<double[]> trajectory		= new ArrayList<double[]>();
		
	//Provide an equals method
	public abstract boolean equals(Object o);
	
	//Provide a hashing method
	public abstract int hash();
	
	//Merge two fits together
	public abstract void merge(Fit newFit);
	
	/*Store fit information. recordFit must ensure that the final position 
	 * VECTOR is also stored, regardless of the internal representation of the 
	 * parameters. */
	public abstract void recordFit(int fitSteps, int functionCalls, 
			double fitTime, double trainLikelihood, Model input); 
	
	//Return final position
	public double[] positionVector() {
		return Array.clone(finalPosition);
	}
	
	/*Record cross-validation likelihoods. These values are averaged per read
	 * AND across the number of folds. */
	public void recordCrossValidate(double testL, double adjustedTestL) {
		this.testLikelihood			= testL;
		this.adjustedTestLikelihood	= adjustedTestL;
		isCrossValidate				= true;
	}
	
	//Store the shift index
	public void recordShiftIndex(int shiftPos) {
		testShifts 		= true;
		this.shiftPos	= shiftPos;
	}
	
	/* Store error bars. recordErrorBars must ensure that the final error  
	 * VECTOR is also stored, regardless of the internal representation of the 
	 * parameters. Also, the number of null vectors must be stored. */
	public abstract void recordErrorBars(int nullVectors, double[] input);
	
	//Store Hessian
	public void storeHessian(Model.CompactGradientOutput input) {
		if(input==null || input.hessian==null) {
			return;
		}
		hessian = Array.clone(input.hessian);
	}
	
	//Print Hessian
	public void printHessian() {
		if (hessian!=null) {
			Array.print(hessian);
		} else {
			System.out.println("No Hessian to Print!");
		}
	}
	
	//Create Hessian file
	public void printHessian(String filePath) {
		try {
			PrintStream outputFile	= new PrintStream(
					new FileOutputStream(filePath));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			printHessian();
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create trajectory file at this "
					+ "location: "+filePath);
			e.printStackTrace();
		}
	}
	
	//Store position update
	public void addStep(double[] positionVector) {
		trajectory.add(positionVector);
	}
	
	//Print fit trajectory
	public void printTrajectories() {
		for (int i=0; i<trajectory.size(); i++) {
			Array.print(trajectory.get(i));
		}
	}
	
	//Print Trajectory File
	public void printTrajectories(String filePath) {
		try {

			PrintStream outputFile	= new PrintStream(
					new FileOutputStream(filePath, false));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			printTrajectories();
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create trajectory file at this "
					+ "location: "+filePath);
			e.printStackTrace();
		}
	}
	
	public void printTrajectories(String filePath, boolean append) {
		if(append == false || trajectory.size() <= 10 ) //Rewrites the full file the first couple of iterations.
			printTrajectories(filePath);
		else {
			try {
				PrintStream outputFile	= new PrintStream(
						new FileOutputStream(filePath, true));
				PrintStream original	= System.out;
				
				System.setOut(outputFile);
				Array.print(trajectory.get(trajectory.size()-1));
				System.setOut(original);
				
			} catch (FileNotFoundException e) {
				System.out.println("Cannot create trajectory file at this "
						+ "location: "+filePath);
				e.printStackTrace();
			}
		}
	}
	
}