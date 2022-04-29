package model;

import base.Fit;
import base.Model;
import base.Array;
import base.Sequence;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.UUID;

public class MultinomialFit extends Fit implements Serializable, Comparable<MultinomialFit> {
	private static final long serialVersionUID = 1183024776966944474L;
	public boolean isDinuc, isShape, isFlank, isNSBinding, isError, isMulti, isSeeded;
	public int k = 0, nCount, nShapeClasses, flankLength = 0;
	public double lambda;
	public String nucSymmetry, dinucSymmetry, saveTime;
	public int[] ks								= null;
	public double[] seed						= null;
	public double[] errors						= null;
	public String[] shapes						= null;
	public String[] nucSymmetries, dinucSymmetries;
	public int[][][] betaIdx					= null;
	public UUID fitID;
	private double maxLikelihood;
	
	public MultinomialFit(Model input, double[] seed) {
		if (input instanceof MultinomialModel) {
			MultinomialModel model = (MultinomialModel) input;
			
			isDinuc			= model.isDinuc();
			isShape			= model.isShape();
			isFlank			= model.isFlank();
			isNSBinding		= model.isNSBinding();
			if (isFlank)	flankLength	= model.getFlankLength();
			k				= model.getK();
			nCount			= model.getNCount();
			lambda			= model.getLambda();
			maxLikelihood	= model.maxLikelihood();
			if (model.getShapeModel()!=null) {
				shapes			= model.getShapeModel().getShapeClasses();
				nShapeClasses	= model.getNShapeClasses();
			} 
			nucSymmetry		= model.getNucSymmetry();
			dinucSymmetry	= model.getDinucSymmetry();
			this.seed		= seed;
			if (seed!=null) {
				isSeeded	= true;
			}
		} else if (input instanceof MultiModeModel) {
			MultiModeModel model = (MultiModeModel) input;
			
			isMulti			= true;
			isDinuc			= model.isDinuc();
			isShape			= model.isShape();
			isFlank			= model.isFlank();
			isNSBinding		= model.isNSBinding();
			if (isFlank)	flankLength	= model.getFlankLength();
			ks				= Array.clone(model.getKs());
			if (ks.length==1) {
				k			= ks[0];
			}
			betaIdx			= Array.clone(model.getBetaIdx());
			nCount			= model.getNCount();
			lambda			= model.getLambda();
			maxLikelihood	= model.maxLikelihood();
			if (model.getShapeModel()!=null) {
				shapes			= model.getShapeModel().getShapeClasses();
				nShapeClasses	= model.getNShapeClasses();
			} 
			nucSymmetries	= Array.clone(model.getNucSymmetries());
			dinucSymmetries	= Array.clone(model.getDinucSymmetries());
			this.seed		= seed;
			if (seed!=null) {
				isSeeded		= true;
			}
		} else {
			throw new IllegalArgumentException("MultinomialFit does not support general a general model class.");
		}
		
		//Common
		DateFormat dateFormat	= new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date date				= new Date();
		fitID 					= UUID.randomUUID();
		saveTime				= dateFormat.format(date);
	}
	
	@Override
	public int compareTo(MultinomialFit o) {
	    final int BEFORE= -1;
	    final int EQUAL = 0;
	    final int AFTER = 1;
	    boolean thisFit, thatFit;
	    int thisSum, oSum, tComp;

	    //this optimization is usually worthwhile, and can always be added
	    if (this == o) return EQUAL;

	    //Single mode fits come before MultiMode
	    if (this.isMulti && !o.isMulti)	return AFTER;
	    if (!this.isMulti && o.isMulti) return BEFORE;
	    
	    //Lowest values of k come first, or smaller models, or smaller overall k
	    if (isMulti) {
	    	if (this.ks.length < o.ks.length)	return BEFORE;
	    	if (this.ks.length > o.ks.length)	return AFTER;
	    	
	    	thisSum = 0;
	    	oSum = 0;
	    	for (int i=0; i<ks.length; i++) {
	    		thisSum += ks[i];
	    		oSum	+= o.ks[i];
	    	}
	    	if (thisSum < oSum)	return BEFORE;
	    	if (thisSum > oSum) return AFTER;
	    } else {
	    	if (this.k < o.k)	return BEFORE;
	    	if (this.k > o.k)	return AFTER;
	    }
	    
	    //Next, lowest values of flank length come first 
	    if (this.flankLength < o.flankLength)	return BEFORE;
	    if (this.flankLength > o.flankLength)	return AFTER;
	    
	    //Next, lowest values of lambda come first
	    if (this.lambda < o.lambda)	return BEFORE;
	    if (this.lambda > o.lambda) return AFTER;
	    
	    //Next, the smallest number of features come first
	    if (this.isNSBinding && !o.isNSBinding)	return AFTER;	//This has NSBinding while o does not, so after
	    if (!this.isNSBinding && o.isNSBinding) return BEFORE;	//This does not have NSBinding features while o does, so before
	    if (this.isDinuc && !o.isDinuc)			return AFTER;	//This has dinuc features while o does not, so after
	    if (!this.isDinuc && o.isDinuc)			return BEFORE;	//This does not have dinuc features while o does, so before
	    if (this.isShape && !o.isShape)			return AFTER;	//This has shape while o does not, so after
	    if (!this.isShape && o.isShape)			return BEFORE;	//This does not have shape, while o does so before
	    if (this.isShape && o.isShape) {						//Both have shape, so sort by number of shape features used
	    	if (this.shapes.length < o.shapes.length)	return BEFORE;
	    	if (this.shapes.length > o.shapes.length)	return AFTER;
	    	
	    	//Have the same number of shape features. Now order them based on content
	    	thisFit = Arrays.asList(this.shapes).contains("MGW");
	    	thatFit	= Arrays.asList(o.shapes).contains("MGW");
	    	if (thisFit && !thatFit)			return BEFORE;	//If this has MGW, it should come before the one being compared
	    	if (!thisFit && thatFit)			return AFTER;	//vice versa
	    	thisFit = Arrays.asList(this.shapes).contains("ProT");
	    	thatFit	= Arrays.asList(o.shapes).contains("ProT");
	    	if (thisFit && !thatFit)			return BEFORE;	//If this has ProT, it should come before the one being compared
	    	if (!thisFit && thatFit)			return AFTER;	//vice versa
	    	thisFit = Arrays.asList(this.shapes).contains("HelT");
	    	thatFit	= Arrays.asList(o.shapes).contains("HelT");
	    	if (thisFit && !thatFit)			return BEFORE;	//If this has HelT, it should come before the one being compared
	    	if (!thisFit && thatFit)			return AFTER;	//vice versa
	    	thisFit = Arrays.asList(this.shapes).contains("Roll");
	    	thatFit	= Arrays.asList(o.shapes).contains("Roll");
	    	if (thisFit && !thatFit)			return BEFORE;	//If this has Roll, it should come before the one being compared
	    	if (!thisFit && thatFit)			return AFTER;	//vice versa
	    }
	    
	    //Next, compare symmetry or the existence of it
	    if (isMulti) {
	    	if (this.nucSymmetries!=null) {
	    		if (o.nucSymmetries==null) {
	    			return AFTER;				//This has nuc symmetry while o does not, so it comes after
	    		}
	    	} else {
	    		if (o.nucSymmetries!=null) {
	    			return BEFORE;				//This does not have nuc symmetry while o does, so it comes before
	    		}
	    	}
	    } else {
		    if (this.nucSymmetry!=null) {
		    	if (o.nucSymmetry!=null) {
		    		tComp = nucSymmetry.compareTo(o.nucSymmetry);
		    		if (tComp!=0) {
		    			return tComp;
		    		}
		    	} else {
		    		return AFTER;				//This has nuc symmetry while o does not, so it comes after
		    	}
		    } else {
		    	if (o.nucSymmetry!=null) {		//This does not have nuc symmetry while o does, so it comes before
		    		return BEFORE;
		    	}
		    }
		    if (this.dinucSymmetry!=null) {
		    	if (o.dinucSymmetry!=null) {
		    		tComp = dinucSymmetry.compareTo(o.dinucSymmetry);
		    		if (tComp!=0) {
		    			return tComp;
		    		}
		    	} else {
		    		return AFTER;				//This has dinuc symmetry while o does not, so it comes after
		    	}
		    } else {
		    	if (o.dinucSymmetry!=null) {	//This does not have dinuc symmetry while o does, so it comes before
		    		return BEFORE;
		    	}
		    }
	    }
	    
	    //Last, compare shifts. Lowest shift first, with no shifts at the lowest position
	    thisSum = (this.testShifts) ? this.shiftPos : Integer.MIN_VALUE;
	    oSum	= (o.testShifts) ? o.shiftPos : Integer.MIN_VALUE;
	    if (thisSum < oSum)	return BEFORE;
	    if (thisSum > oSum) return AFTER;
	    
		return EQUAL;
	}
	
	@Override
	public boolean equals(Object o) {
		if (this==o)	return true;
		if (!(o instanceof MultinomialFit))	return false;
		MultinomialFit check = (MultinomialFit) o;
		
		//First check to see if both are Single or MultiMode fits
		if (isMulti != check.isMulti) {
			return false;
		}
		
		//Next check feature types
		if (this.isShape && check.isShape) {
			if (this.shapes.length==check.shapes.length) {
				for (int i=0; i<this.shapes.length; i++) {
					if (!this.shapes[i].equals(check.shapes[i])) {
						return false;
					}
				}
			} else {
				return false;
			}
		} else if((this.isShape && !check.isShape) || (!this.isShape && check.isShape)) {
			return false;
		}
		
		//Next check symmetries and modes
		if (isMulti) {
			if (this.ks.length != check.ks.length) {
				return false;
			}
			//Check to see if the kmer fitting is taking place in the same order
			//Path dependence is important for some techniques
			for (int i=0; i<ks.length; i++) {
				if (ks[i] != check.ks[i]) {
					return false;
				}
			}
			if ((this.nucSymmetries==null && check.nucSymmetries!=null) ||
				(this.nucSymmetries!=null && check.nucSymmetries==null) ||
				(this.dinucSymmetries==null && check.dinucSymmetries!=null) ||
				(this.dinucSymmetries!=null && check.dinucSymmetries==null)) {
				return false;
			}
			for (int i=0; i<ks.length; i++) {
				if (this.nucSymmetries!=null && !nucSymmetries[i].equals(check.nucSymmetries[i])) {
					return false;
				}
				if (this.dinucSymmetries!=null && !dinucSymmetries[i].equals(check.dinucSymmetries[i])) {
					return false;
				}
			}
		} else {
			if (this.nucSymmetry!=null && !this.nucSymmetry.equals(check.nucSymmetry)) {
				return false;
			} else if(this.nucSymmetry==null && check.nucSymmetry!=null) {
				return false;
			}
			if (this.dinucSymmetry!=null && !this.dinucSymmetry.equals(check.dinucSymmetry)) {
				return false;
			} else if(this.dinucSymmetry==null && check.dinucSymmetry!=null) {
				return false;
			}
			if (this.k != check.k) {
				return false;
			}
		}
		
	    return 
	    		this.lambda		== check.lambda &&
	    		this.flankLength== check.flankLength &&
	    		this.isDinuc	== check.isDinuc &&
	    		this.isNSBinding== check.isNSBinding &&
	    		this.testShifts == check.testShifts &&
	    		this.shiftPos	== check.shiftPos;
	}
	
	public int hash() {		//Computes hash for all numerical entries
		return	Arrays.deepHashCode(hessian) + Arrays.hashCode(seed) + 
				Arrays.hashCode(ks) + Arrays.hashCode(finalPosition) + 
				Arrays.hashCode(errors);
	}
 	
	public void merge(Fit in) {		//merge two fits together
		MultinomialFit newFit;
		if (in instanceof MultinomialFit) {
			newFit = (MultinomialFit) in;
		} else {
			throw new IllegalArgumentException("merge requires an object of type MultinomialFit!");
		}
		if (newFit.isMulti!=isMulti) {
			throw new IllegalArgumentException("Attempting to merge two fits of different models!");
		}
		if (newFit.finalPosition!=null && Arrays.hashCode(finalPosition)!=Arrays.hashCode(newFit.finalPosition)) {
			finalPosition = Array.clone(newFit.finalPosition);
		}
		if (newFit.errors!=null && Arrays.hashCode(errors)!=Arrays.hashCode(newFit.errors)) {
			errors = Array.clone(newFit.errors);
		}
		if (newFit.isSeeded && Arrays.hashCode(seed)!=Arrays.hashCode(newFit.seed)) {
			seed = Array.clone(newFit.seed);
		}
		if (newFit.hessian!=null && Arrays.hashCode(hessian)!=Arrays.hashCode(newFit.hessian)) {
			hessian = Array.clone(hessian);
		}
	}
	
	public void recordFit(int fitSteps, int functionCalls, double fitTime, 
			double trainLikelihood, Model input) {
		if (input instanceof MultinomialModel || input instanceof MultiModeModel) {
			finalPosition			= Array.clone(input.getPositionVector());
		} else {
			throw new IllegalArgumentException("MultinomialFit does not support general a general model class.");
		}
		//Common
		this.fitSteps			= fitSteps;
		this.functionCalls		= functionCalls;
		this.fitTime			= fitTime;
		this.trainLikelihood	= trainLikelihood;
	}
	
	public void recordErrorBars(int nullVectors, double[] input) {
		if (input==null) {			//Handle the case where there is no hessian
			return;
		}
		isError			= true;
		this.nullVectors= nullVectors;
		errors			= Array.clone(input);
	}
	
	public void print(boolean isPSAM, boolean isSeed, boolean isRaw) {
		System.out.println("-------------------");
		if (testShifts)	System.out.println("Shift Index:\t\t"+shiftPos);
		System.out.print("k:\t\t\t\t\t");
		if (isMulti) {
			System.out.print(ks[0]);
			for (int i=1; i<ks.length; i++) {
				System.out.print(","+ks[i]);
			}
		} else {
			System.out.print(k);
		}
		if (isFlank)	System.out.print("\tFlank: "+flankLength);
		if (lambda!=0)	System.out.print("\tL2 \u03BB: "+lambda);
		System.out.print("\nFeatures Used:\t\tNucleotide");
		if (isDinuc)	System.out.print(", Dinucleotide");
		if (isShape)	System.out.print(", Shape");
		if (isNSBinding)System.out.print(", NSBinding");
		if (isShape) {
			System.out.print("; Shapes Used: ");
			Array.print(shapes);
		} else {
			System.out.print("\n");
		}
		if (nucSymmetry!=null)	System.out.println("Nuc Symmetry:\t\t"+nucSymmetry);
		if (nucSymmetries!=null) {
			System.out.print("Nuc Symmetry:\t\t"+nucSymmetries[0]);
			for (int i=1; i<ks.length; i++) {
				System.out.print(" || "+nucSymmetries[i] );
			}
		}
		if (dinucSymmetry!=null && isDinuc) {
			System.out.println("Dinuc Symmetry:\t\t"+dinucSymmetry);
		}
		if (dinucSymmetries!=null) {
			System.out.print("\nDinuc Symmetry:\t\t"+dinucSymmetries[0]);
			for (int i=1; i<ks.length; i++) {
				System.out.print(" || "+dinucSymmetries[i] );
			}
		}
		System.out.print("\nRun Time:\t\t");
		System.out.printf("%10.3f",fitTime);
		System.out.print("s\tFitting Steps: "+fitSteps+"\tFunction Calls: "+functionCalls+"\n");		
		System.out.println("Train Likelihood:\t"+trainLikelihood+"\tPer Read: "+
		trainLikelihood/nCount+"\tAdjusted: "+(trainLikelihood-maxLikelihood)/nCount);
		if (isCrossValidate) {
			System.out.println("Test Likelihood:\t"+testLikelihood+"\tAdjusted: "+adjustedTestLikelihood);
		}
		if (isPSAM) {
			if (isMulti) {
				double maxVal, totOffset;
				double[] betas, error = null;
				
				System.out.print("Mode Relative Betas:\t");
				for (int i=0; i<ks.length; i++) {
					totOffset = 0;
					maxVal = Double.NEGATIVE_INFINITY;
					betas  = Arrays.copyOfRange(finalPosition, betaIdx[i][1][0], betaIdx[i][1][1]+1);
					for (int g=0; g<4; g++) {
						maxVal = Math.max(maxVal, betas[g]);
					}
					totOffset += maxVal;
					maxVal = Double.NEGATIVE_INFINITY;
					for (int g=0; g<4; g++) {
						maxVal = Math.max(maxVal, betas[betas.length-g-1]);
					}
					totOffset += maxVal;
					if (i!=0) {
						System.out.printf(", %10.5f", totOffset);
					} else {
						System.out.printf("%10.5f", totOffset);
					}
				}
				if (isNSBinding) {
					if (isError) {
						System.out.printf("\t\tNS Binding:\t%10.4f \u00B1 %-10.4f", 
								finalPosition[finalPosition.length-1], errors[errors.length-1]);
					} else {
						System.out.printf("\t\tNS Binding:\t%10.5f", finalPosition[finalPosition.length-1]);
					}
				}
				System.out.print("\n");
				
				for (int i=0; i<ks.length; i++) {
					System.out.print("\nMode "+(i+1));
					betas = Arrays.copyOfRange(finalPosition, betaIdx[i][0][0], betaIdx[i][0][1]+1);
					//Correct first beta idx
					maxVal = Double.NEGATIVE_INFINITY;
					for (int g=0; g<4; g++) {
						maxVal = Math.max(maxVal, betas[g]);
					}
					for (int g=0; g<4; g++) {
						betas[g] -= maxVal;
					}
					if (isError) {
						error = Arrays.copyOfRange(errors, betaIdx[i][0][0], betaIdx[i][0][1]+1);
					}
					printPSAM(betas, error, ks[i]);
				}
			} else {
				printPSAM(finalPosition, errors, k);
			}
		}
		if (isError)System.out.println("\nNumber of Null Vectors:\t"+nullVectors);
		if (isSeed && isSeeded) {
			System.out.println("\nSeed: ");
			Array.print(seed);
		}
		if (isRaw) {
			System.out.println("\nRaw Beta Values: ");
			Array.print(finalPosition);
		}
	}
	
	public void print(String filePath, boolean isPSAM, boolean isSeed, boolean isRaw) {
		try {
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			print(isPSAM, isSeed, isRaw);
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create properties file at this location: "+filePath);
			e.printStackTrace();
		}
	}
	
	private double[] extractNuc(double[] input, int k) {
		return Arrays.copyOfRange(input, 0, 4*k);
	}
	
	private double[] extractDinuc(double[] input, int k) {
		return Arrays.copyOfRange(input, 4*k, 4*k+16*(k-1));
	}
	
	private double[] extractShape(double[] input, int k) {
		int offset = (isDinuc) ? 4*k + 16*(k-1) : 4*k;
		return Arrays.copyOfRange(input, offset, offset+nShapeClasses*k);
	}
	
	private double extractNS(double[] input) {
		return input[input.length-1];
	}
	
	public void printPSAM(double[] betas, double[] errors, int k){
		int colIdx, rowIdx;
		double maxVal, nsBinding, nsBindingError = 0;
		int[] maxNucs		= new int[k];
		double[] nucBetas	= null, nucError = null, dinucBetas = null;
		double[] dinucError	= null, shapeBetas = null, shapeError = null;
		double[][] nucPSAM	= new double[4][k];
		double[][] errorPSAM= new double[4][k];
		String currBase;
		
		//Extract Features
		nucBetas	= extractNuc(betas, k);
		dinucBetas	= (isDinuc) ? extractDinuc(betas, k) : null;
		shapeBetas	= (isShape)	? extractShape(betas, k) : null;
		nsBinding	= (isNSBinding) ? extractNS(betas) : 0;
		if (isError) {
			nucError		= extractNuc(errors, k);
			dinucError		= (isDinuc) ? extractDinuc(errors, k) : null;
			shapeError		= (isShape)	? extractShape(errors, k) : null;
			nsBindingError	= (isNSBinding) ? extractNS(errors) : 0;
		}
		
		//Nucleotide Features
		System.out.print("\n");
		for (int i=0; i<nucBetas.length; i++) {				//create PSAM Matrix
			colIdx = i/4;
			rowIdx = i%4;
			nucPSAM[rowIdx][colIdx] = nucBetas[i];
			if (isError) {
				errorPSAM[rowIdx][colIdx] = nucError[i];
			}
		}
		for (int col=0; col<k; col++) {						//find top sequence
			maxVal = nucPSAM[0][col];
			rowIdx = 0;
			for (int row=1; row<4; row++) {
				if (nucPSAM[row][col] > maxVal){
					maxVal = nucPSAM[row][col];
					rowIdx = row;
				}
			}
			maxNucs[col] = rowIdx;
		}
		System.out.print("Nucleotide PSAM:\t");				//print top sequence
		for (int i=0; i<k; i++){
			currBase = (new Sequence(maxNucs[i], 1)).getString();
			System.out.print(currBase);
		}
		//NS Binding
		if (isNSBinding && !isMulti) {
			if (isError) {
				System.out.printf("\t\tNS Binding:\t%10.4f \u00B1 %-10.4f", nsBinding, nsBindingError);
			} else {
				System.out.printf("\t\tNS Binding:\t%10.5f", nsBinding);
			}
		}
		System.out.print("\n");

		for (int i=0; i<k; i++) {
			currBase = (new Sequence(maxNucs[i], 1)).getString();
			System.out.print(" "+currBase+"  ");
			for (int row=0; row<4; row++) {
				if (isError) {
					System.out.printf("%10.4f \u00B1 %-10.4f", nucPSAM[row][i], errorPSAM[row][i]);
				} else {
					System.out.printf("%10.5f  ", nucPSAM[row][i]); 
				}
			}
			System.out.print("   "+currBase+"\n");
		}

		//Dinuc Features
		if (isDinuc) {
			double[][] dinucPSAM	= new double[16][k-1];
			errorPSAM				= new double[16][k-1];
			
			for (int i=0; i<dinucBetas.length; i++) {			//create PSAM Matrix
				colIdx = i/16;
				rowIdx = i%16;
				dinucPSAM[rowIdx][colIdx] = dinucBetas[i];
				if (isError) {
					errorPSAM[rowIdx][colIdx] = dinucError[i];
				}
			}
			System.out.println("\nDinucleotide PSAM:");
			for (int i=0; i<k-1; i++) {
				System.out.printf("%2d  ",i);
				for (int row=0; row<16; row++) {
					if (isError) {
						System.out.printf("%10.4f \u00B1 %-10.4f", dinucPSAM[row][i], errorPSAM[row][i]);
					} else {
						System.out.printf("%10.5f  ", dinucPSAM[row][i]); 
					}
				}
				System.out.printf("  %-2d",i);
				System.out.print("\n");
			}
		}
		
		//Shape Features
		if (isShape) {
			double[][] shapePSAM	= new double[nShapeClasses][k];
			errorPSAM				= new double[nShapeClasses][k];
			
			for (int i=0; i<shapeBetas.length; i++) {			//create PSAM Matrix
				colIdx = i/nShapeClasses;
				rowIdx = i%nShapeClasses;
				shapePSAM[rowIdx][colIdx] = shapeBetas[i];
				if (isError) {
					errorPSAM[rowIdx][colIdx] = shapeError[i];
				}
			}
			System.out.println("\nShape PSAM:");
			for (int i=0; i<k; i++) {
				currBase = (new Sequence(maxNucs[i], 1)).getString();
				System.out.print(" "+currBase+"  ");
				for (int row=0; row<nShapeClasses; row++) {
					if (isError) {
						System.out.printf("%10.4f \u00B1 %-10.4f", shapePSAM[row][i], errorPSAM[row][i]);
					} else {
						System.out.printf("%10.5f  ", shapePSAM[row][i]); 
					}
				}
				System.out.print("   "+currBase+"\n");
			}
		}
	}
	
	public void printList(int i, boolean isCSV) {
		boolean isHit;
		int colIdx 			= 0;
		int rowIdx 			= 0;
		double maxVal, nsBinding;
		int[] maxNucs		= new int[k];
		double[] b, e, s;
		double[][] nucPSAM	= new double[4][k];
		String currBase;
		
		nsBinding = (isNSBinding) ? extractNS(finalPosition) : 0;		
		if (isCSV) {
			if (isMulti) {
				System.out.print(i+","+saveTime+","+fitTime+","+fitSteps+","+functionCalls+",NA,"+flankLength);
			} else {
				System.out.print(i+","+saveTime+","+fitTime+","+fitSteps+","+functionCalls+","+k+","+flankLength);
			}
			if (isNSBinding) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (isDinuc) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (isShape && Arrays.asList(shapes).contains("MGW")) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (isShape && Arrays.asList(shapes).contains("ProT")) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (isShape && Arrays.asList(shapes).contains("HelT")) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (isShape && Arrays.asList(shapes).contains("Roll")) {
				System.out.print(",TRUE");
			} else {
				System.out.print(",FALSE");
			}
			if (nucSymmetry!=null) {
				System.out.print(","+nucSymmetry.replace(",", "."));
			} else {
				System.out.print(",NA");
			}
			if (dinucSymmetry!=null && isDinuc) {
				System.out.print(","+dinucSymmetry.replace(",", "."));
			} else {
				System.out.print(",NA");
			}
			if (testShifts) {
				System.out.print(","+shiftPos);
			} else {
				System.out.print(",NA");
			}
			System.out.print(","+nsBinding+","+trainLikelihood+","+(trainLikelihood/nCount)+","+
					((trainLikelihood-maxLikelihood)/nCount));
			if (isCrossValidate) {
				System.out.print(","+testLikelihood+","+adjustedTestLikelihood+",");
			} else {
				System.out.print(",NA,NA,");
			}
			if (lambda==0) {
				System.out.print("0,");
			} else {
				System.out.print(lambda+",");
			}
			if (isError) {
				System.out.print(nullVectors+",");
			} else {
				System.out.print("NA,");
			}
			
			//Print PSAM if not multimode
			if (isMulti) {
				System.out.print("NA");
			} else {
				for (int j=0; j<4*k; j++) {				//create PSAM Matrix
					colIdx = j/4;
					rowIdx = j%4;
					nucPSAM[rowIdx][colIdx] = extractNuc(finalPosition, k)[j];
				}
				for (int col=0; col<k; col++) {						//find top sequence
					maxVal = nucPSAM[0][col];
					rowIdx = 0;
					for (int row=1; row<4; row++) {
						if (nucPSAM[row][col] > maxVal){
							maxVal = nucPSAM[row][col];
							rowIdx = row;
						}
					}
					maxNucs[col] = rowIdx;
				}
				for (int j=0; j<k; j++){
					currBase = (new Sequence(maxNucs[j], 1)).getString();
					System.out.print(currBase);
				}
			}
			//Print raw values
			if (isMulti) {
				for (int j=0; j<ks.length; j++) {
					b = Arrays.copyOfRange(finalPosition, betaIdx[j][0][0], betaIdx[j][0][1]+1);
					e = (isError) ? Arrays.copyOfRange(errors, betaIdx[j][0][0], betaIdx[j][0][1]+1) : null;
					s = (isSeeded) ? Arrays.copyOfRange(seed, betaIdx[j][0][0], betaIdx[j][0][1]+1) : null;
					listString(b, e, s, ks[j], false);
				}
				if (isNSBinding) {
					System.out.print(",NSB>");
					System.out.print(","+finalPosition[finalPosition.length-1]);
					if (isError) {
						System.out.print(",NSBE>");
						System.out.print(","+errors[errors.length-1]);
					}
					if (isSeeded) {
						System.out.print(",NSBS>");
						System.out.print(","+seed[seed.length-1]);
					}
				}
			} else {
				listString(finalPosition, errors, seed, k, true);
			}

			System.out.print(",<EOL>\n");
		} else {
			if (isMulti && ks.length>1) {
				System.out.printf("%5d\t    -\t%5d ",i,flankLength);
			} else {
				System.out.printf("%5d\t%5d\t%5d ",i,k,flankLength);				
			}
			if (isNSBinding) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (isDinuc) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (isShape && Arrays.asList(shapes).contains("MGW")) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (isShape && Arrays.asList(shapes).contains("ProT")) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (isShape && Arrays.asList(shapes).contains("HelT")) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (isShape && Arrays.asList(shapes).contains("Roll")) {
				System.out.print(" XX ");
			} else {
				System.out.print("    ");
			}
			if (!isMulti && nucSymmetry!=null) {
				System.out.print(" XX ");
			} else if (isMulti && nucSymmetries!=null) {
				isHit = false;
				for (String currSymm : nucSymmetries) {
					if (currSymm!="null") {
						isHit = true;
						break;
					}
				}
				if (isHit) {
					System.out.print(" XX ");
				} else {
					System.out.print("    ");
				}
			} else {
				System.out.print("    ");
			}
			if (!isMulti && dinucSymmetry!=null) {
				System.out.print(" XX ");
			} else if (isMulti && isDinuc && nucSymmetries!=null) {
				isHit = false;
				for (String currSymm : dinucSymmetries) {
					if (currSymm!="null") {
						isHit = true;
						break;
					}
				}
				if (isHit) {
					System.out.print(" XX ");
				} else {
					System.out.print("    ");
				}
			} else {
				System.out.print("    ");
			}
			if (testShifts && (!isMulti || ks.length==1)) {
				System.out.printf(" %5d", shiftPos);
			} else {
				System.out.print("     -");
			}
			if (isNSBinding) {
				System.out.printf("%10.4f", nsBinding);
			} else {
				System.out.print("          ");				
			}
			if (isCrossValidate) {
				System.out.printf("%10.4f%10.4f", (trainLikelihood/nCount), testLikelihood);
			} else {    
				System.out.printf("%10.4f          ", (trainLikelihood/nCount));
			}
			if (isError) {
				System.out.printf("%10.4f     %5d   ", lambda, nullVectors);
			} else {
				System.out.printf("%10.4f         -   ", lambda);
			}
		
			//Print PSAM if not a multi-mode model
			if (isMulti && ks.length>1) {
				System.out.print(" --- ");
			} else {
				int currK;
				if (isMulti) {
					currK = ks[0];					
				} else {
					currK = k;
				}
				for (int j=0; j<4*currK; j++) {				//create PSAM Matrix
					colIdx = j/4;
					rowIdx = j%4;
					nucPSAM[rowIdx][colIdx] = extractNuc(finalPosition, currK)[j];
				}
				for (int col=0; col<currK; col++) {						//find top sequence
					maxVal = nucPSAM[0][col];
					rowIdx = 0;
					for (int row=1; row<4; row++) {
						if (nucPSAM[row][col] > maxVal){
							maxVal = nucPSAM[row][col];
							rowIdx = row;
						}
					}
					maxNucs[col] = rowIdx;
				}
				for (int j=0; j<currK; j++){
					currBase = (new Sequence(maxNucs[j], 1)).getString();
					System.out.print(currBase);
				}
			}

			System.out.print("\t\t"+saveTime+"\n");
		}
	}
	
	private void listString(double[] betas, double[] error, double[] seed, int k, boolean printNS) {
		double[] currArray;
		
		System.out.print(",NB>");
		currArray = extractNuc(betas, k);
		for (int j=0; j<currArray.length; j++) {
			System.out.print(","+currArray[j]);
		}
		if (isDinuc) {
			System.out.print(",DB>");
			currArray = extractDinuc(betas, k);
			for (int j=0; j<currArray.length; j++) {
				System.out.print(","+currArray[j]);
			}
		}
		if (isShape) {
			System.out.print(",SB>");
			currArray = extractShape(betas, k);
			for (int j=0; j<currArray.length; j++) {
				System.out.print(","+currArray[j]);
			}
		}
		if (isNSBinding && printNS) {
			System.out.print(",NSB>");
			System.out.print(","+betas[betas.length-1]);
		}
		if (isError) {
			System.out.print(",NE>");
			currArray = extractNuc(error, k);
			for (int j=0; j<currArray.length; j++) {
				System.out.print(","+currArray[j]);
			}
			if (isDinuc) {
				System.out.print(",DE>");
				currArray = extractDinuc(error, k);
				for (int j=0; j<currArray.length; j++) {
					System.out.print(","+currArray[j]);
				}
			}
			if (isShape) {
				System.out.print(",SE>");
				currArray = extractShape(error, k);
				for (int j=0; j<currArray.length; j++) {
					System.out.print(","+currArray[j]);
				}
			}
			if (isNSBinding && printNS) {
				System.out.print(",NSBE>");
				System.out.print(","+error[error.length-1]);
			}
		}	
		if (isSeeded) {
			System.out.print(",NS>");
			currArray = extractNuc(seed, k);
			for (int j=0; j<currArray.length; j++) {
				System.out.print(","+currArray[j]);
			}
			if (isDinuc) {
				System.out.print(",DS>");
				currArray = extractDinuc(seed, k);
				for (int j=0; j<currArray.length; j++) {
					System.out.print(","+currArray[j]);
				}
			}
			if (isShape) {
				System.out.print(",SS>");
				currArray = extractShape(seed, k);
				for (int j=0; j<currArray.length; j++) {
					System.out.print(","+currArray[j]);
				}
			}
			if (isNSBinding && printNS) {
				System.out.print(",NSBS>");
				System.out.print(","+seed[seed.length-1]);
			}
		}
	}
}