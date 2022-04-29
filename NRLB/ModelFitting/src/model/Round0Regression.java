package model;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import config.ExperimentReference;
import base.ArraysMergerHeap;
import base.CountObject;
import base.MarkovModelInfo;
import base.MarkovModelOption;
import main.SELEX;
import main.SimpleKmerCount;
import minimizers.*;
import utils.FileUtilities;
import base.Sequence;

import base.Array;

public class Round0Regression {
	public static int l;
	//Markov Model Parameters
	public static float[] modelProbabilities 	= null;
	public static int[] modelCounts 		= null;
	public static long modelTotalCount		= 0;
	public static String modelObjPath;
		
	public static void main(String[] args) {
		/**
		 * Configuration
		 * Please build Markov Model and count table using the Build_MM_Kmer_Table file
		 */
		//Sample info
		String selexWorkingDir		= "F:/Data/Datasets/tmp/";
		String r0ConfigFile		= "./config/p53_Mut.xml";
		String sequencingRunName	= "p53-Acetylated-248Q-R0";
		String testSampleName		= "R0";
		String trainSampleName		= "R0";
		l				= 26;
		String lFlankTest		= "GGTAGTGGAGGTGGGACTGT";	//left flanking sequence
		String rFlankTest		= "ACAGTGAGGTGGAGTAGG";		//right flanking sequence
		String lFlankTrain		= "GGTAGTGGAGGTGGGACTGT";
		String rFlankTrain		= "ACAGTGAGGTGGAGTAGG";
		boolean isSplit			= true;
		//Markov Model Information
		boolean useMarkovModel		= false;
		int modelLength			= 6;
		String modelMethod		= MarkovModelOption.DIVISION;
		//Fitting Parameters
		String minimizerType		= "LBFGS";
		boolean lbfgsMCSearch		= false;
		int lbfgsMem			= 200;
		int lbfgsMaxIters		= 2500;
		double lbfgsConvergence		= 5E-8;
		boolean psRandomAxis		= false;
		double psInitStep		= .5;
		double psTheta			= .5;
		double psConvergence		= 1E-12;
		boolean[] testFlanks		= {true};
		int[] testKs			= {1, 2, 3, 4};
		double[] seed			= null;
		//Output
		boolean isVerbose		= true;
		String outputPath		= "F:/Data/";
		String dataFileName		= "p53_Mut";
		String trajectoryFile		= null;

		if (args.length>0) {
			if (args.length!=7 && args.length!=8) {
				throw new IllegalArgumentException("Incorrect number of arguments supplied!");
			}
			//Sample and output config
			l			= Integer.parseInt(args[0]);
			selexWorkingDir		= args[1];
			sequencingRunName	= args[2]+".0";
			testSampleName		= "R0";
			trainSampleName		= "R0";
			lFlankTest		= args[3];
			rFlankTest		= args[4];
			lFlankTrain		= args[3];
			rFlankTrain		= args[4];
			dataFileName		= args[2];
			outputPath		= args[5];
			isVerbose		= false;
			trajectoryFile		= null;
			//Predefined Markov model definition
			useMarkovModel		= false;
			//Predefined minimizer configuration
			minimizerType		= "LBFGS";
			lbfgsMCSearch		= false;
			lbfgsMem		= 200;
			lbfgsMaxIters		= 5000;
			lbfgsConvergence	= 5E-8;
			//Fit config
			isSplit			= true;
			testFlanks		= new boolean[]{true};
			if (args.length==7) {
				testKs		= new int[]{Integer.parseInt(args[6])};
			} else {
				int inMinK	= Integer.parseInt(args[6]);
				int inMaxK	= Integer.parseInt(args[7]);
				testKs		= new int[inMaxK-inMinK+1];
				int currIntIdx	= 0;
				for (int currInK=inMinK; currInK<=inMaxK; currInK++) {
					testKs[currIntIdx] = currInK;
					currIntIdx++;
				}
			}
			seed			= null;
		}

		/**
		 * Runtime Variables
		 */
		double likelihood, adjL, mmLikelihood = 0;
		Data[] datasets;
		Minimizer minimizer;
		Round0Model train, test;
		Round0Fit currFit			= null;
		Round0Results results	  	= null;
		FileUtilities reader		= new FileUtilities(l);
		ExperimentReference trainingSet, testingSet;
		String trainFilePath		= selexWorkingDir+sequencingRunName+"."+trainSampleName+".0."+l+".dat";
		String testFilePath			= selexWorkingDir+sequencingRunName+"."+testSampleName+".0."+l+".dat";
		
		//Define Results object
		results = new Round0Results(l, minimizerType, psInitStep, psTheta, 
				psConvergence, psRandomAxis, lbfgsMem, lbfgsMaxIters, 
				lbfgsConvergence, lbfgsMCSearch);
			
		//Read in dataset
		if (isSplit) {
			datasets	= reader.readSeqFile(trainFilePath, lFlankTrain, rFlankTrain, isSplit);
			results.defineTrainSet(sequencingRunName, trainSampleName, 
					trainFilePath, datasets[0].nCount, lFlankTrain, rFlankTrain);
			results.defineTestSet(isSplit, sequencingRunName, testSampleName, 
					testFilePath, datasets[2].nCount);
		} else {
			//Load Markov Model + Calculate Likelihood
			if (useMarkovModel) {
				SELEX.setWorkingDirectory(selexWorkingDir);
				SELEX.loadConfigFile(r0ConfigFile);
				trainingSet	= SELEX.getExperimentReference(sequencingRunName, trainSampleName, 0);
				testingSet	= SELEX.getExperimentReference(sequencingRunName, testSampleName, 0);
				loadMarkovModel(trainingSet, testingSet, modelLength, modelMethod);
				mmLikelihood = markovModelLikelihood(sequencingRunName, testSampleName, testFilePath, modelLength);
				results.defineMarkovModel(modelObjPath, mmLikelihood);
			}
			
			datasets	= new Data[2];
			datasets[0] = reader.readSeqFile(trainFilePath, lFlankTrain, rFlankTrain, false)[0];
			datasets[1] = reader.readSeqFile(testFilePath, lFlankTest, rFlankTest, false)[0];
			results.defineTrainSet(sequencingRunName, trainSampleName, 
					trainFilePath, datasets[0].nCount, lFlankTrain, rFlankTrain);
			results.defineTestSet(isSplit, sequencingRunName, testSampleName, 
					testFilePath, datasets[1].nCount);
		}
		System.out.println("Finished reading in data.");
		
		//Find optimal Round0 Model with cross validation
		for (int k : testKs) {
			for (boolean useFlanks : testFlanks) {
				System.out.println("Currently testing model for k = "+k+" and Flank = "+useFlanks);
				try {
					if (isSplit) {
						train	= new Round0Model(datasets[1], k, useFlanks);
						test	= new Round0Model(datasets[2], k, useFlanks);
						if (minimizerType.equals("LBFGS")) {
							minimizer = new LBFGS(train, lbfgsMem, lbfgsConvergence, lbfgsMaxIters, lbfgsMCSearch, false, false, isVerbose);
						} else {
							minimizer = new PatternSearch(train, psInitStep, psConvergence, psTheta, psRandomAxis, false, false, isVerbose);
						}
						while (true) {
							try {
								currFit = (Round0Fit) minimizer.minimize(seed, trajectoryFile)[0];
								break;
							} catch (Exception e) {
								continue;
							}
						}
						test.setParams(currFit.betas);
						likelihood = test.functionEval();
						adjL = (likelihood-test.maxLikelihood())*test.likelihoodNormalizer();
						train.replaceData(datasets[0]);
						while (true) {
							try {
								currFit = (Round0Fit) minimizer.minimize(seed, trajectoryFile)[0];
								break;
							} catch (Exception e) {
								continue;
							}
						}
						currFit.recordCrossValidate(likelihood, adjL);						
					} else {
						train	= new Round0Model(datasets[0], k, useFlanks);
						test	= new Round0Model(datasets[1], k, useFlanks);
						if (minimizerType.equals("LBFGS")) {
							minimizer = new LBFGS(train, lbfgsMem, lbfgsConvergence, lbfgsMaxIters, lbfgsMCSearch, false, false, isVerbose);
						} else {
							minimizer = new PatternSearch(train, psInitStep, psConvergence, psTheta, psRandomAxis, false, false, isVerbose);
						}
						while (true) {
							try {
								currFit = (Round0Fit) minimizer.minimize(seed, trajectoryFile)[0];
								break;
							} catch (Exception e) {
								continue;
							}
						}
						test.setParams(currFit.betas);
						likelihood = test.functionEval();
						adjL = (likelihood-test.maxLikelihood())*test.likelihoodNormalizer();
						currFit.recordCrossValidate(likelihood, adjL);
						if (useMarkovModel) {
							currFit.recordMarkovModel(mmLikelihood-likelihood);
						}
					}
					results.addFit(currFit);
					results.store(outputPath+dataFileName, true);
					results.print(outputPath+dataFileName, true, true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}			
		}
	}
	
	public static void loadMarkovModel(ExperimentReference trainSet, ExperimentReference testSet, 
			int modelLength, String modelMethod) {
		//Load Markov Model
		MarkovModelInfo mm = SELEX.getMarkModelInfo(trainSet, testSet,  modelLength, null, modelMethod);
		//Markov Model properties
		modelObjPath 			= mm.getMarkovObjPath();
		String modelCountPath 	= mm.getMarkovCountsPath();
		modelTotalCount 		= mm.getMarkovLengthTotalCount();

		try {
			System.out.println("Reading Markov Prob file:" + modelObjPath);
			FileInputStream fos = new FileInputStream(modelObjPath);
			ObjectInputStream oos = new ObjectInputStream(fos);
			modelProbabilities = (float[]) oos.readObject();

			System.out.println("Reading Markov Count file:" + modelCountPath);
			ArraysMergerHeap.MinHeapNode mmNode = new ArraysMergerHeap.MinHeapNode(modelCountPath);
			CountObject obj = null;
			modelCounts = new int[1 << (2 * (modelLength))];

			while ((obj = mmNode.peek()) != null) {
				Sequence seq = (Sequence) obj.getKey();
				modelCounts[(int) seq.getValue()] = obj.getCount();
				mmNode.pop();
			}
			oos.close();
		} catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		System.out.println("Finished loading Markov Model.");
	}
	
	public static double markovModelLikelihood(String seqRunName, String testSampleName, 
			String testFilePath, int modelLength) {
		double likelihood 	= 0;
		double Z 			= 0;
		int currCount 		= 0;
		int n 				= 0;
		Sequence currSeq;
		CountObject obj 	= null;
		ArraysMergerHeap.MinHeapNode node = new ArraysMergerHeap.MinHeapNode(testFilePath);
		
		//Calculate partition function
		for (long sequence = 0; sequence < Math.pow(4, l); sequence++) {
			Z += SimpleKmerCount.getPredictedCount(modelLength, modelTotalCount, 
					new Sequence(sequence, l), modelCounts, modelProbabilities);
		}
		Z = Math.log(Z);
		//Calculate -Log Likelihood
		while ((obj = node.peek()) != null) {
			currCount 	= obj.getCount();
			currSeq		= obj.getKey();
			n += currCount;
			likelihood -= currCount * Math.log(SimpleKmerCount.getPredictedCount(modelLength, modelTotalCount,
					currSeq, modelCounts, modelProbabilities));
			node.pop();
		}
		likelihood += n * Z;
		System.out.println("Fitness for Length "+modelLength+" Markov Model");
		System.out.println("Z: "+Z);
		System.out.println("-Log Likelihood: "+likelihood);
		return likelihood;
	}
}
