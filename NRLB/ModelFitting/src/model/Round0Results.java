package model;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;

import base.Fit;
import base.Results;

public class Round0Results extends Results {
	private static final long serialVersionUID = 6431639090078976338L;
	public boolean psRandomAxis, lbfgsMCSearch, isSplit, useMarkovModel;
	public int l, lbfgsMem, lbfgsMaxIters, trainCounts, testCounts;
	public double psInitStep, psTheta, psConvergence, lbfgsConvergence;
	public double mmLikelihood;
	public String minimizerType	= null;
	public String sequencingRunName = "", trainSampleName = "", trainSamplePath = "";
	public String testSampleName = "", testSamplePath = "", lFlank = "", rFlank ="";
	public String markovModelFileName = "", timeStamp = "";
	public ArrayList<Round0Fit> fits;
	
	public Round0Results(String fileName) {
		try {
			read(fileName);
		} catch (Exception e) {
			System.err.println(fileName + " could not be read!");
			e.printStackTrace();
		}
	}
	
	public Round0Results(int l, String minimizerType, double psInitStep, double psTheta, double psConvergence, boolean psRandomAxis, 
			int lbfgsMem, int lbfgsMaxIters, double lbfgsConvergence, boolean lbfgsMCSearch) {
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
		fits = new ArrayList<Round0Fit>();
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		timeStamp = dateFormat.format(date);
	}
	
	public void defineTrainSet(String sequencingRunName, String trainSampleName, 
			String trainSamplePath, int trainCounts, String lFlank, String rFlank){
		this.sequencingRunName	= sequencingRunName;
		this.trainSampleName	= trainSampleName;
		this.trainSamplePath	= trainSamplePath;
		this.trainCounts		= trainCounts;
		this.lFlank				= lFlank;
		this.rFlank				= rFlank;
	}
	
	public void defineTestSet(boolean isSplit, String sequencingRunName, String testSampleName, String testSamplePath, int testCounts) throws IllegalArgumentException{
		this.isSplit 			= isSplit;
		if (isSplit) {
			this.testSampleName = trainSampleName;
		} else {
			if (!this.sequencingRunName.equals(sequencingRunName)) {
				throw new IllegalArgumentException("Sequencing Run Name of the test dataset does not match that of the training dataset.");
			}
			this.testSampleName	= testSampleName;			
		}
		this.testSamplePath 	= testSamplePath;
		this.testCounts			= testCounts;
	}
	
	public void defineMarkovModel(String markovModelFileName, double mmLikelihood) {
		this.markovModelFileName	= markovModelFileName;
		this.mmLikelihood			= mmLikelihood;
		useMarkovModel				= true;
	}

	@Override
	public void addFit(Fit in) {
		Round0Fit fit, currFit;
		if (in instanceof Round0Fit) {
			fit = (Round0Fit) in;
		} else {
			throw new IllegalArgumentException("Need MultinomialFit");
		}

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
	}
	
	public Round0Fit getBestFit() {
		int bestIdx = 0;
		double bestLik = fits.get(0).testLikelihood;
		
		for (int i=1; i<fits.size(); i++) {
			if (fits.get(i).testLikelihood<bestLik) {
				bestIdx = i;
			}
		}
		
		return fits.get(bestIdx);
	}
	
	public Round0Fit getFit(int k, boolean isFlank) {
		int searchHits	= 0;
		int hitIdx		= 0;
		Round0Fit currFit;
		
		for (int i=0; i<fits.size(); i++) {
			currFit = fits.get(i);
			if ((currFit.k == k) && (currFit.isFlank == isFlank)) {
				hitIdx = i;
				searchHits++;
			}
		}
		if (searchHits==0){
			throw new IllegalArgumentException("No fit exists for k="+k+" and flank= "+isFlank);
		} else if(searchHits>1) {
			throw new IllegalArgumentException("Multiple fit matches found.");
		} else {
			return fits.get(hitIdx);
		}
	}
	
	@Override
	public void read(String fileName) throws Exception {
		FileInputStream fis			= new FileInputStream(fileName);
		ObjectInputStream ois		= new ObjectInputStream(fis);
		Round0Results fitResult		= (Round0Results) ois.readObject();
		ois.close();
		
		psRandomAxis		= fitResult.psRandomAxis;
		lbfgsMCSearch		= fitResult.lbfgsMCSearch;
		l					= fitResult.l;
		trainCounts			= fitResult.trainCounts;
		testCounts			= fitResult.testCounts;
		lbfgsMem			= fitResult.lbfgsMem;
		lbfgsMaxIters		= fitResult.lbfgsMaxIters;
		psInitStep			= fitResult.psInitStep;
		psTheta				= fitResult.psTheta;
		psConvergence		= fitResult.psConvergence;
		lbfgsConvergence	= fitResult.lbfgsConvergence;
		mmLikelihood		= fitResult.mmLikelihood;
		lFlank				= fitResult.lFlank;
		rFlank				= fitResult.rFlank;
		minimizerType		= fitResult.minimizerType;
		isSplit				= fitResult.isSplit;		
		useMarkovModel		= fitResult.useMarkovModel;
		sequencingRunName	= fitResult.sequencingRunName;
		testSampleName		= fitResult.testSampleName;
		testSamplePath		= fitResult.testSamplePath;
		trainSampleName		= fitResult.trainSampleName;
		trainSamplePath		= fitResult.trainSamplePath;
		markovModelFileName	= fitResult.markovModelFileName;
		timeStamp			= fitResult.timeStamp;
		fits				= fitResult.fits;
	}

	@Override
	public boolean equals(Object compareTo) {
		if (this==compareTo)	return true;
		if (!(compareTo instanceof Round0Results))	return false;
		Round0Results check = (Round0Results) compareTo;
		
		return 
				this.psRandomAxis		== check.psRandomAxis &&
				this.lbfgsMCSearch		== check.lbfgsMCSearch &&
				this.l					== check.l &&
				this.trainCounts		== check.trainCounts &&
				this.testCounts			== check.testCounts &&
				this.lbfgsMem			== check.lbfgsMem &&
				this.lbfgsMaxIters		== check.lbfgsMaxIters &&
				this.psInitStep			== check.psInitStep && 
				this.psTheta			== check.psTheta &&
				this.psConvergence		== check.psConvergence && 
				this.lbfgsConvergence	== check.lbfgsConvergence &&
				this.mmLikelihood		== check.mmLikelihood &&
				this.lFlank.equals(check.lFlank) && 
				this.rFlank.equals(check.rFlank) &&
				this.minimizerType.equals(check.minimizerType) && 
				this.isSplit			== check.isSplit &&
				this.sequencingRunName.equals(check.sequencingRunName) &&
				this.testSamplePath.equals(check.testSamplePath) &&
				this.testSampleName.equals(check.testSampleName) &&
				this.trainSamplePath.equals(check.trainSamplePath) &&
				this.trainSampleName.equals(check.trainSampleName) &&
				this.markovModelFileName.equals(check.markovModelFileName);
	}

	@Override
	public void store(String fileName, boolean isAppend) throws Exception {
		DateFormat dateFormat	= new SimpleDateFormat("yyyy-MM-dd-HH-mm");
		Date date				= new Date();
		File f					= new File(fileName+".dat");
		
		if(isAppend) {							//Append to new file?
			if (f.isFile()) {					//If so, check to see if a file already exists
				Round0Results oldFitResult = new Round0Results(fileName+".dat");	//If exists, read it in
				//Consistency Checks: If it passed, merge files
				if(this.equals(oldFitResult)) {
					timeStamp = oldFitResult.timeStamp;		//reflect old timestamp
					//Add fits from old file into new one safely (using .addFit() from above)
					for (Round0Fit currFit : oldFitResult.fits) {
						this.addFit(currFit);
					}
				} else {		//Did not pass equality check. Create new file name
					fileName = fileName+"_"+dateFormat.format(date);
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

	public void print(boolean isSeed, boolean isRaw) {
		System.out.println("*******************");
		System.out.println("* FIT INFORMATION *");
		System.out.println("*******************");
		System.out.println("Sequencing Run Name:    "+sequencingRunName);
		System.out.println("Training Sample Name:   "+trainSampleName+"\tCounts: "+trainCounts);
		System.out.println("Training File Location: "+trainSamplePath);
		System.out.println("Testing Sample Name:    "+testSampleName+"\tCounts: "+testCounts);
		System.out.println("Testing File Location:  "+testSamplePath);
		if (useMarkovModel) {
			System.out.println("Markov Model Used:      "+markovModelFileName);
			System.out.println("Markov Model Likelihood:"+mmLikelihood);
		}
		System.out.println("Length:                 "+l);
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
		for (Round0Fit currFit : fits) {
			currFit.print(isSeed, isRaw);
		}
	}
	
	public void print(String filePath, boolean isSeed, boolean isRaw) {
		try {
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath+".txt"));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			print(isSeed, isRaw);
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create properties file at this location: "+filePath);
			e.printStackTrace();
		}
	}
	
	public void printList() {
		System.out.println("Index,Time,FitTime,FitSteps,FncCalls,k,isFlank,"+
				"TrainL,TrainLPerRead,AdjTrainLPerRead,"+
				"TestLPerRead,AdjTestLPerRead,MarkovModelDiff");
		for (int i=0;  i<fits.size(); i++) {
			fits.get(i).printCSV(i);
		}
	}
	
	public void printList(String filePath, boolean isCSV) {
		try {
			filePath				= (isCSV) ? filePath+".csv" : filePath+".list";
			PrintStream outputFile	= new PrintStream(new FileOutputStream(filePath));
			PrintStream original	= System.out;
			
			System.setOut(outputFile);
			printList();
			System.setOut(original);
		} catch (FileNotFoundException e) {
			System.out.println("Cannot create fit list file at this location: "+filePath);
			e.printStackTrace();
		}
	}
}
