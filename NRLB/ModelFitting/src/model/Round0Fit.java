package model;

import base.Fit;
import base.Model;
import base.Array;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.UUID;

public class Round0Fit extends Fit implements Serializable, Comparable<Round0Fit> {
	private static final long serialVersionUID = -5735644133197523114L;
	public boolean isFlank, isMarkovModel;
	public int k, nCount;
	public String saveTime;
	public double markovModelDiff;
	public double[] betas				= null;
	public double[] seed				= null;
	public UUID fitID;
	private double maxLikelihood;
	
	public Round0Fit(Model input, double[] seed) {
		Round0Model model;
		if (input instanceof Round0Model) {
			model = (Round0Model) input;
		} else {
			throw new IllegalArgumentException("Round0Fit does not support general a general model class.");
		}
		
		DateFormat dateFormat	= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date date				= new Date();
		
		isFlank			= model.isFlank();
		k				= model.getK();
		nCount			= model.getNCount();
		maxLikelihood	= model.maxLikelihood();
		if (seed!=null) {
			seed		= Array.clone(seed);
		}
		fitID 			= UUID.randomUUID();
		saveTime		= dateFormat.format(date);
	}
	
	@Override
	public int compareTo(Round0Fit o) {
	    final int BEFORE= -1;
	    final int EQUAL = 0;
	    final int AFTER = 1;

	    //this optimization is usually worthwhile, and can always be added
	    if (this == o) return EQUAL;
	    
	    //Lowest values of k come first
	    if (this.k < o.k)	return BEFORE;
	    if (this.k > o.k)	return AFTER;
	    
	    //Next, no flank comes first 
	    if (this.isFlank && !o.isFlank)	return AFTER;
	    if (!this.isFlank && o.isFlank)	return BEFORE;
	    
		return EQUAL;
	}
	
	@Override
	public boolean equals(Object o) {
		if (this==o)	return true;
		if (!(o instanceof Round0Fit))	return false;
		Round0Fit check = (Round0Fit) o;
		
	    return 
	    		this.k			== check.k &&
	    		this.isFlank	== check.isFlank;
	}
	
	public int hash() {		//Computes hash for all numerical entries
		return 
				Arrays.hashCode(betas) + Arrays.hashCode(seed);
	}
	
	public void merge(Fit in) {		//merge two fits together
		Round0Fit newFit;
		if (in instanceof Round0Fit) {
			newFit = (Round0Fit) in;
		} else {
			throw new IllegalArgumentException("Merge requires an object of type Round0Fit!");
		}
		
		if (newFit.betas!=null && Arrays.hashCode(betas)!=Arrays.hashCode(newFit.betas)) {
			betas = Array.clone(newFit.betas);
		}
		if (newFit.seed!=null && Arrays.hashCode(seed)!=Arrays.hashCode(newFit.seed)) {
			seed = Array.clone(newFit.seed);
		}
	}
	
	public void recordFit(int fitSteps, int functionCalls, double fitTime, 
			double trainLikelihood, Model input) {
		Round0Model model;
		if (input instanceof Round0Model) {
			model = (Round0Model) input;
		} else {
			throw new IllegalArgumentException("Round0Fit does not support general a general model class.");
		}
		
		this.fitSteps			= fitSteps;
		this.functionCalls		= functionCalls;
		this.fitTime			= fitTime;
		this.trainLikelihood	= trainLikelihood;
		betas	 				= model.getPositionVector();
	}
	
	public void recordErrorBars(int nullVectors, double[] input) {
		throw new IllegalArgumentException("Round0Model does not compute error bars.");
	}
	
	public void recordMarkovModel(double input) {
		markovModelDiff = input;
		isMarkovModel	= true;
	}
	
	public void print(boolean isSeed, boolean isRaw) {
		System.out.println("-------------------");
		System.out.print("k:\t"+k);
		if (isFlank)	System.out.print("\tFlank=TRUE");
		System.out.print("\nRun Time:\t\t");
		System.out.printf("%10.3f",fitTime);
		System.out.print("s\tFitting Steps: "+fitSteps+"\tFunction Calls: "+functionCalls+"\n");		
		System.out.println("Train Likelihood:\t"+trainLikelihood+"\tPer Read: "+
		trainLikelihood/nCount+"\tAdjusted: "+(trainLikelihood-maxLikelihood)/nCount);
		if (isCrossValidate) {
			System.out.println("Test Likelihood:\t"+testLikelihood+"\tAdjusted: "+adjustedTestLikelihood);
		}
		if (isMarkovModel) {   
			System.out.println("Diff vs. M Model:  	"+markovModelDiff);
		}
		if (isSeed) {
			if (seed!=null){
				System.out.println("\nSeeding Information:");
				Array.print(seed);
			}
		}
		if (isRaw) {
			System.out.println("\nRaw Beta Values:");
			Array.print(betas);
		}
	}
	
	public void print(String filePath, boolean isSeed, boolean isRaw) {
		try {
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath));
			PrintStream original	= System.out;
	
			System.setOut(outputFile);
			print(isSeed, isRaw);
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create properties file at this location: "+filePath);
			e.printStackTrace();
		}
	}

	public void printCSV(int i) {
		System.out.print(i+","+saveTime+","+fitTime+","+fitSteps+","+functionCalls+","+k+",");
		if (isFlank) {
			System.out.print(",TRUE");
		} else {
			System.out.print(",FALSE");
		}
		System.out.print(","+trainLikelihood+","+(trainLikelihood/nCount)+","+
				((trainLikelihood-maxLikelihood)/nCount));
		if (isCrossValidate) {
			System.out.print(","+testLikelihood+","+adjustedTestLikelihood+",");
		} else {
			System.out.print(",NA,NA,");
		}
		if (isMarkovModel) {
			System.out.print(","+markovModelDiff+",");
		} else {
			System.out.print(",NA,");
		}
		System.out.print(",B>");
		for (int j=0; j<betas.length; j++) {
			System.out.print(","+betas[j]);
		}
		if (seed!=null) {
			System.out.print(",S>");
			for (int j=0; j<seed.length; j++) {
				System.out.print(","+seed[j]);
			}
		}
		System.out.print(",<EOL>\n");
	}
}