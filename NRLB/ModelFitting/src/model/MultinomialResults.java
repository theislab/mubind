package model;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;

import base.Results;
import base.Fit;
import base.Sequence;

public class MultinomialResults extends Results implements Serializable {
	private static final long serialVersionUID = -7262249074565477766L;
	//Fit Parameters
	public boolean psRandomAxis, lbfgsMCSearch, crossValidate, filterReads;
	public int l, nFolds, maxBaseCount, maxBaseRepeat, removedReads;
	public int totCounts				= 0;
	public int lFlankLength				= 0;
	public int rFlankLength				= 0;
	public int lbfgsMem, lbfgsMaxIters;
	public double psInitStep, psTheta, psConvergence;
	public double lbfgsConvergence;
	public double testRatio;
	public long leftFlank				= 0;
	public long rightFlank				= 0;
	public String minimizerType			= null;
	public String shapeDir				= "";
	public String lFlank				= "";
	public String rFlank				= "";
	public String sequencingRunName		= "";
	public String sampleName			= "";
	public String samplePath			= "";
	public String configFile			= "";
	public String timeStamp;
	public String[] regex				= null;
	//Round0 Parameters
	public int r0k						= 0;
	public boolean r0Flank				= false;
	public String r0ModelPath			= "";
	public ArrayList<MultinomialFit> fits;
	transient private boolean autoSave	= false;
	transient private String autoSaveLocation;
	
	public MultinomialResults(String fileName) {
		try {
			read(fileName);
		} catch (Exception e) {
			System.err.println(fileName + " could not be read!");
			e.printStackTrace();
		}
	}
	
	public MultinomialResults(int l, String minimizerType, double psInitStep, double psTheta, double psConvergence, boolean psRandomAxis, 
			int lbfgsMem, int lbfgsMaxIters, double lbfgsConvergence, boolean lbfgsMCSearch, String configFile) {
		this.l					= l;
		this.minimizerType		= minimizerType;
		this.psInitStep			= psInitStep;
		this.psTheta			= psTheta;
		this.psConvergence		= psConvergence;
		this.psRandomAxis		= psRandomAxis;
		this.lbfgsMem			= lbfgsMem;
		this.lbfgsMaxIters		= lbfgsMaxIters;
		this.lbfgsConvergence	= lbfgsConvergence;
		this.lbfgsMCSearch		= lbfgsMCSearch;
		this.configFile			= configFile;
		fits = new ArrayList<MultinomialFit>();
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		timeStamp = dateFormat.format(date);
	}
	
	public void autoSave(String filePath) {
		autoSave = true;
		autoSaveLocation = filePath;
	}
	
	public void defineR0Model(Round0Model R0Model) {
		r0k			= R0Model.getK(); 
		r0Flank		= R0Model.isFlank();
		r0ModelPath	= R0Model.modelPath();
	}
	
	public void defineShapeModel(String shapeDir) {
		this.shapeDir	= shapeDir;
	}
	
	public void defineDataset(String sequencingRunName, String sampleName, String samplePath, 
			String lFlank, String rFlank, int totCounts, boolean crossValidate, double testRatio, int nFolds) {
		this.sequencingRunName		= sequencingRunName;
		this.sampleName				= sampleName;
		this.samplePath				= samplePath;
		this.lFlank					= lFlank;
		lFlankLength				= lFlank.length();
		leftFlank					= (new Sequence(lFlank, 0, lFlankLength)).getValue();
		this.rFlank					= rFlank;
		rFlankLength				= rFlank.length();
		rightFlank					= (new Sequence(rFlank, 0, rFlankLength)).getValue();
		this.totCounts				= totCounts;
		this.crossValidate			= crossValidate;
		this.testRatio				= testRatio;
		this.nFolds					= nFolds;
	}
	
	public void defineFilter(int maxBaseCount, int maxBaseRepeat, int removedReads, String[] regex) {
		this.filterReads			= true;
		this.maxBaseCount			= maxBaseCount;
		this.maxBaseRepeat			= maxBaseRepeat;
		this.removedReads			= removedReads;
		this.regex					= regex;
	}
	
	public void addFit(Fit in) {
		MultinomialFit fit;
		if (in instanceof MultinomialFit) {
			fit = (MultinomialFit) in;
		} else {
			throw new IllegalArgumentException("Need MultinomialFit");
		}
		addFit(fit, false);
	}
	
	public void addFit(MultinomialFit fit, boolean ignoreAutoSave) {
		MultinomialFit currFit; 
		
		//First ensure fit is unique
		for (int i=0; i<fits.size(); i++) {
			currFit = fits.get(i);
			if (currFit.equals(fit)) {
				//Check UUID
				if (currFit.fitID.equals(fit.fitID)) {
					//UUID is the same, so check to see if data stored is the same
					if (currFit.hash()==fit.hash()) {
						//fit is exactly the same as currFit, ignore add operation
						return;
					} else {
						//fit contains new data, so merge the two
						fits.get(i).merge(fit);
						return;
					}
				} else {
					//UUID is different. Overwrite current fit
					fits.set(i, fit);
					return;
				}
			}
		}
		fits.add(fit);
		Collections.sort(fits);
		
		if (autoSave && !ignoreAutoSave) {
			try {
				store(autoSaveLocation, true);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
		
	public MultinomialFit getFit(int idx) {
		return fits.get(idx);
	}
	
	public Fit getBestFit() {
		boolean isHit			= false;
		int currBestIdx			= 0;
		double minLikelihood	= Double.MAX_VALUE;
		
		for (int i=0; i<fits.size(); i++) {
			if (crossValidate) {
				if (fits.get(i).testLikelihood < minLikelihood) {
					minLikelihood = fits.get(i).testLikelihood;
					currBestIdx = i;
					isHit = true;
				}
			} else {
				if (fits.get(i).testLikelihood < minLikelihood) {
					minLikelihood = fits.get(i).trainLikelihood;
					currBestIdx = i;
					isHit = true;
				}				
			}
		}
		
		if (!isHit) {
			throw new IllegalArgumentException("No fits have been stored!");
		} else {
			return fits.get(currBestIdx);
		}
	}
	
	public int nFits() {
		return fits.size();
	}
	
	//TODO: Fix this section as needed
	public void read(String fileName) throws Exception{
		FileInputStream fis		= new FileInputStream(fileName);
		ObjectInputStream ois	= new ObjectInputStream(fis);
		MultinomialResults fitResult = (MultinomialResults) ois.readObject();
		ois.close();
		
		psRandomAxis		= fitResult.psRandomAxis;
		lbfgsMCSearch		= fitResult.lbfgsMCSearch;
		crossValidate		= fitResult.crossValidate;
		l					= fitResult.l;
		nFolds				= fitResult.nFolds;
		totCounts			= fitResult.totCounts;
		lFlankLength		= fitResult.lFlankLength;
		rFlankLength		= fitResult.rFlankLength;
		lbfgsMem			= fitResult.lbfgsMem;
		lbfgsMaxIters		= fitResult.lbfgsMaxIters;
		psInitStep			= fitResult.psInitStep;
		psTheta				= fitResult.psTheta;
		psConvergence		= fitResult.psConvergence;
		lbfgsConvergence	= fitResult.lbfgsConvergence;
		testRatio			= fitResult.testRatio;
		leftFlank			= fitResult.leftFlank;
		rightFlank			= fitResult.rightFlank;
		minimizerType		= fitResult.minimizerType;
		shapeDir			= fitResult.shapeDir;
		lFlank				= fitResult.lFlank;
		rFlank				= fitResult.rFlank;
		sequencingRunName	= fitResult.sequencingRunName;
		sampleName			= fitResult.sampleName;
		samplePath			= fitResult.samplePath;
		configFile			= fitResult.configFile;
		timeStamp			= fitResult.timeStamp;
		r0k					= fitResult.r0k;
		r0Flank				= fitResult.r0Flank;
		r0ModelPath			= fitResult.r0ModelPath;
		fits				= fitResult.fits;
		filterReads			= fitResult.filterReads;
		maxBaseCount		= fitResult.maxBaseCount;
		maxBaseRepeat		= fitResult.maxBaseRepeat;
		removedReads		= fitResult.removedReads;
		regex				= fitResult.regex;
	}
	
	//TODO: Fix Section as needed
	@Override
	public boolean equals(Object compareTo) {							//WARNING: Config File is NOT part of Equals check
		if (this==compareTo)	return true;
		if (!(compareTo instanceof MultinomialResults))	return false;
		MultinomialResults check = (MultinomialResults) compareTo;

		if (this.crossValidate == check.crossValidate) {				//Both have the same CV status
			if (this.crossValidate) {
				return (this.nFolds == check.nFolds && this.testRatio == check.testRatio);
			} else {
				return true;
			}
		} 																//Else ignore
		
		return 
				this.psRandomAxis		== check.psRandomAxis &&
				this.lbfgsMCSearch		== check.lbfgsMCSearch &&
				this.l					== check.l &&
				this.totCounts			== check.totCounts &&
				this.lbfgsMem			== check.lbfgsMem &&
				this.lbfgsMaxIters		== check.lbfgsMaxIters &&
				this.psInitStep			== check.psInitStep && 
				this.psTheta			== check.psTheta &&
				this.psConvergence		== check.psConvergence && 
				this.lbfgsConvergence	== check.lbfgsConvergence &&
				this.minimizerType.equals(check.minimizerType) && 
				this.shapeDir.equals(check.shapeDir) &&
				this.lFlank.equals(check.lFlank) && 
				this.rFlank.equals(check.rFlank) &&
				this.sequencingRunName.equals(check.sequencingRunName) &&
				this.sampleName.equals(check.sampleName) &&
				this.samplePath.equals(check.samplePath) &&
				this.r0k				== check.r0k && 
				this.r0Flank			== check.r0Flank && 
				this.r0ModelPath.equals(check.r0ModelPath) &&
				this.filterReads		== check.filterReads &&
				this.maxBaseCount		== check.maxBaseCount &&
				this.maxBaseRepeat		== check.maxBaseRepeat &&
				this.regex.equals(check.regex);
	}
	
	public void store(String fileName, boolean isAppend) throws Exception{
		DateFormat dateFormat	= new SimpleDateFormat("yyyy-MM-dd-HH-mm");
		Date date				= new Date();
		File f					= new File(fileName+".dat");
		
		if(isAppend) {							//Append to new file?
			if (f.isFile()) {					//If so, check to see if a file already exists
				MultinomialResults oldFitResult = new MultinomialResults(fileName+".dat");	//If exists, read it in
				//Consistency Checks: If it passed, merge files
				if(this.equals(oldFitResult)) {
					configFile= oldFitResult.configFile;	//reflect old config file
					timeStamp = oldFitResult.timeStamp;		//reflect old timestamp
					if (this.crossValidate==false && oldFitResult.crossValidate==true) {
						this.crossValidate	= true;
						this.nFolds			= oldFitResult.nFolds;
						this.testRatio		= oldFitResult.testRatio;
					}
					//Add fits from old file into new one safely (using .addFit() from above)
					for (MultinomialFit currFit : oldFitResult.fits) {
						this.addFit(currFit, true);
					}
				} else {		//Did not pass equality check. Create new file name
					fileName = fileName+"_"+dateFormat.format(date);
					if (autoSave) {
						autoSaveLocation = fileName;
					}
				}
			}
		} else if (f.isFile()) {	//If not append, see if file exists. If it does, create new file name
			fileName = fileName+"_"+dateFormat.format(date);
		}
		
		//Write
		FileOutputStream fos 	= new FileOutputStream(fileName+".dat");
		ObjectOutputStream oos 	= new ObjectOutputStream(fos);	
		oos.writeObject(this);
		oos.close();
	}
	
	//TODO: Fix as needed
	public void print(boolean isPSAM, boolean isSeed, boolean isRaw) {
		System.out.println("*******************");
		System.out.println("* FIT INFORMATION *");
		System.out.println("*******************");
		System.out.println("Configuration File:     "+configFile);
		System.out.println("Sequencing Run Name:    "+sequencingRunName);
		System.out.println("Sample Name:            "+sampleName);
		System.out.println("Sample File Location:   "+samplePath);
		System.out.print("Filter Reads:           "+filterReads);
		if (filterReads) {
			System.out.print("\t\tMaximum Allowed Base Count: "+maxBaseCount+"\t\tMaximum Repeat Length: "+maxBaseRepeat+"\t\tRegex: ");
			if (regex!=null) {
				for (int i=0; i<regex.length; i++) {
					System.out.print(regex[i]+"\t");
				}
			}
		}
		System.out.print("\nTotal Counts:           "+totCounts);
		if (filterReads) {
			System.out.print("\t\tRemoved Reads: "+removedReads+"\t\t\tLength: "+l+"\n");
		} else {
			System.out.print("\t\tLength: "+l+"\n");
		}
		System.out.print("Use Cross Validation?   "+crossValidate);
		if (crossValidate) {
			System.out.print("\t\tTest Ratio: "+testRatio+"\t\tn-Fold Averaging: "+nFolds);
		}
		System.out.println("\nRound0 Length:          "+r0k+"\t\tUse Flanks? "+r0Flank);
		System.out.println("Round0 Model Location:  "+r0ModelPath);
		if (shapeDir != "") {
			System.out.println("Shape Model Location:   "+shapeDir);	
		}
		System.out.println("Left Flank Sequence:    "+lFlank);
		System.out.println("Right Flank Sequence:   "+rFlank);
		System.out.print("Minimizer Type:         "+minimizerType);
		if (minimizerType.equals("LBFGS")) {
			if (lbfgsMCSearch) {
				System.out.print(" with MC Search\n");
			} else {
				System.out.print(" with Line Search\n");
			}
			System.out.println("Memory Depth:           "+lbfgsMem+"\t\tMaximum Iterations: "+lbfgsMaxIters+
					"\t\t\tConvergence Criteria: "+lbfgsConvergence);
		} else {
			System.out.print("\tRandom Axis Search: "+psRandomAxis+"\n");
			System.out.println("Initial Step Size: "+psInitStep+"\tTheta: "+psTheta+"\tConvergence Criteria: "+psConvergence);
		}
		System.out.println("Fit Time:               "+timeStamp);
		System.out.println("*******************");
		System.out.println("*   FIT RESULTS   *");
		System.out.println("*******************");
		for (MultinomialFit currFit : fits) {
			currFit.print(isPSAM, isSeed, isRaw);
		}
	}
	
	public void print(String filePath, boolean isPSAM, boolean isSeed, boolean isRaw) {
		try {
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath+".txt"));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			print(isPSAM, isSeed, isRaw);
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create properties file at this location: "+filePath);
			e.printStackTrace();
		}
	}
	
	public void printList(boolean isCSV) {
		if (isCSV) {
			System.out.println("Index,Time,FitTime,FitSteps,FncCalls,k,Flank,NS,Di,MG,"+
					"PT,HT,RO,NuSym,DiSym,Shift,NSBind,TrainL,TrainLPerRead,AdjTrainLPerRead,"+
					"TestLPerRead,AdjTestLPerRead,Lambda,NullVectors,PSAM");
		} else {
			System.out.println("LIST OF FIT RESULTS");
			System.out.println("                               ----Shapes----  -Symm-");
			System.out.println("Index\t    k\tFlank  NS  Di  MG  PT  HT  RO  Nu  Di  Shift   NS Bind   Train L    Test L    Lambda  NullVecs   PSAM");	
		}
		for (int i=0;  i<fits.size(); i++) {
			fits.get(i).printList(i, isCSV);
		}
	}
	
	public void printList(String filePath, boolean isCSV) {
		try {
			filePath				= (isCSV) ? filePath+".csv" : filePath+".list";
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			printList(isCSV);
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create fit list file at this location: "+filePath);
			e.printStackTrace();
		}
	}
}