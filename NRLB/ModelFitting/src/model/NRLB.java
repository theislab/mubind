package model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

import base.*;
import minimizers.*;
import utils.FileUtilities;

public class NRLB{
	static boolean isVerbose, crossValidate, isR0Flank, testShifts, hasSymmetry, useDinuc; 
	static boolean psRandomAxis, lbfgsMCSearch, storeHessian, errorBars, isSeeded;
	static boolean printPSAM, printSeeds, printRaw, filterReads, growDinuc, smSweep;
	static int nThreads, l, lbfgsMem, lbfgsMaxIters, nShifts;
	static int nFolds, maxBaseCount, maxBaseRepeat, nModes;
	static double testRatio, psInitStep, psTheta, psConvergence;
	static double lambda = 0, lbfgsConvergence;
	static String configFile, sequencingRunName, sampleName, samplePath, lFlank;
	static String rFlank, minimizerType, shapeDir, R0ModelPath, outputLocation;
	static String outputDataName;
	static boolean[] useNSBinding, useSymmetry;
	static int[] flankLengths, testKs;
	static double[] nsBindingSeed;
	static String[] regexFilter;
	static ArrayList<String> config;
	static ArrayList<int[]> startK = null, maxK = null;
	static ArrayList<double[]> modeSeeds = null;
	static ArrayList<String[]> shapes;
	
	public static void main(String[] args) {
		boolean isShape, isFlank, hitMax;
		int currFlank, currK;
		String nucSymmetry = null, dinucSymmetry = null;
		boolean[] activeModes;
		int[] currKs = null, sK = null, mK = null;
		double[] pos, seed;
		String[] nucSym = null, dinucSym = null;
		Shape shapeModel;
		Round0Model R0Model;
		MultiModeModel multiModel;
		MultinomialModel mmModel;
		FileUtilities reader;
		Fit[] currFits				= null;
		MultinomialResults results	= null;
		ArrayList<Object[]> datasets;
		MultinomialFit latestFit;
		
		//Read and process config file
		if (args.length<1) {	//Has a configuration file been provided?
			throw new IllegalArgumentException("No configuration file defined!");
		}
		configFile = args[0];
		try {
			readConfigFile(configFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//Define objects
		R0Model	= new Round0Model(R0ModelPath, l, -1, isR0Flank, lFlank, rFlank);
		results	= new MultinomialResults(l, minimizerType, psInitStep, psTheta, psConvergence, psRandomAxis,
				lbfgsMem, lbfgsMaxIters, lbfgsConvergence, lbfgsMCSearch, configFile);
		reader	= new FileUtilities(l);
		datasets= reader.readSeqFile(samplePath, R0Model, lFlank, rFlank, 
				crossValidate, nFolds, testRatio, filterReads, maxBaseCount,
				maxBaseRepeat, regexFilter);
		
		results.defineDataset(sequencingRunName, sampleName, samplePath, lFlank, 
				rFlank, ((Data) datasets.get(0)[0]).nCount, crossValidate, testRatio, nFolds);
		if (filterReads)	results.defineFilter(maxBaseCount, maxBaseRepeat, reader.removedReads, regexFilter);
		results.defineR0Model(R0Model);
		if (shapeDir!=null)	results.defineShapeModel(shapeDir);
		results.autoSave(outputLocation+"/"+outputDataName);
		
		//Perform regressions over all available option combinations
		for (String[] currShape : shapes) {								//Loop over shape features
			if (currShape[0].equals("0")) {
				isShape		= false;
				shapeModel	= null;
			} else {
				isShape		= true;
				shapeModel	= new Shape(shapeDir, true);
				shapeModel.setFeatures(currShape);
			}
			for (int flankLength : flankLengths) {					//Loop over flanks	
				isFlank		= (flankLength==0) ? false : true;
				System.out.println("Testing flank length "+flankLength);
				for (boolean isNSBinding : useNSBinding) {		//Loop over NS Binding 
					//Modes have been seeded; need to perform mode-regression first
					if (isSeeded) {
						//Test to see if flank length + kmer lengths to be fit are valid
						if (l+2*flankLength-Array.max(startK.get(0))+1<1 || l+2*flankLength > 32 ||
								(isShape && l+2*flankLength+4 > 32)) {
							continue;	
						}
						
						//Build base model and seed
						seed	= null;
						nucSym	= null;
						dinucSym= null;
						for (int i=0; i<nModes; i++) {
							if (hasSymmetry) {
								if (useSymmetry[i]) {
									nucSym = Array.cat(nucSym, evenOdd(startK.get(0)[i]));
									if (useDinuc) {
										dinucSym = Array.cat(dinucSym, evenOdd(startK.get(0)[i]-1));
									}
								} else {
									nucSym = Array.cat(nucSym, "null");
									if (useDinuc)	dinucSym = Array.cat(dinucSym, "null");
								}
							}
							seed = Array.cat(seed, modeSeeds.get(i));
						}
						if (isNSBinding) {
							seed = Array.cat(seed, 0);
						}
						//First, perform mode-mode regression
						multiModel = new MultiModeModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), isFlank, flankLength,
									useDinuc, isShape, isNSBinding, startK.get(0), nucSym, dinucSym);
						multiModel.setLambda(lambda*multiModel.getNCount());
						multiModel.setParams(seed);
						
						//Do NOT run mode-mode regression if there is only one mode and no NS binding
						if (!(nModes==1 && !isNSBinding)) {
							System.out.println("Performing Mode-Mode regression to resolve relative affinities of the seeds.");
							multiModel.setModeRegression(true);
							currFits = optimizeModel(multiModel, null);
							latestFit= minFit(currFits);
							latestFit.finalPosition = multiModel.getMergedPositionVector();
							multiModel.setModeRegression(false);
							seed = latestFit.finalPosition;
						}
						
						//Now let the modes float
						System.out.println("Switching to regular regression.");
						currFits = optimizeModel(multiModel, seed);						
						latestFit= minFit(currFits);
						latestFit.finalPosition = multiModel.normalize();
						savePrint(latestFit, true, results, true); 
						multiModel.threadPoolShutdown();
					} //Single mode sweep
					else if (smSweep) {						
						for (int k : testKs) {		//Loop over all k's
							//Test to see if flank length and k are larger than 32 bases
							if (l+2*flankLength-k+1<1 || l+2*flankLength > 32 ||
									(isShape && l+2*flankLength+4 > 32)) {
								continue;	
							}
							System.out.println("Fitting NRLB model for length "+k);
							//Symmetry string
							nucSymmetry = (hasSymmetry) ? evenOdd(k) : null;
							dinucSymmetry = (hasSymmetry && useDinuc) ? evenOdd(k-1) : null;

							//Fit a nuc only model first
							mmModel = new MultinomialModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), k, 
									isFlank, flankLength, false, false, isNSBinding, nucSymmetry, null, false);
							mmModel.setLambda(lambda*mmModel.getNCount());
							currFits = optimizeModel(mmModel, datasets, null, testShifts, results);
							mmModel.threadPoolShutdown();
							
							//Add dinucleotides and shape to the best fit if required
							if (isShape || useDinuc) {
								latestFit = minFit(currFits);
								pos = latestFit.finalPosition;
								seed = Arrays.copyOfRange(pos, 0, 4*k);
								seed = (useDinuc) ? Array.cat(seed, new double[16*(k-1)]) : seed;
								seed = (isShape) ? Array.cat(seed, new double[shapeModel.nShapeFeatures()*k]) : seed;
								seed = (isNSBinding) ? Array.cat(seed, pos[pos.length-1]) : seed;
								
								mmModel = new MultinomialModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), k,
										isFlank, flankLength, useDinuc, isShape, isNSBinding, nucSymmetry, dinucSymmetry, false);
								mmModel.setLambda(lambda*mmModel.getNCount());
								currFits = optimizeModel(mmModel, datasets, seed, testShifts, results);
								mmModel.threadPoolShutdown();
							}
							savePrint(null, false, results, true);
						}
					} //Standard fitting
					else {
						//Loop over all kmer combination sets
						for (int currKSet=0; currKSet<startK.size(); currKSet++) {
							sK = startK.get(currKSet);
							mK = maxK.get(currKSet);
							//Test to see if flank + kmer length are valid
							if (l+2*flankLength-Array.max(sK)+1<1 || l+2*flankLength > 32 ||
									(isShape && l+2*flankLength+4 > 32)) {
								continue;	
							}
							System.out.print("Performing sequential fitting for the starting k: ");
							Array.print(sK);
							//Build base model, ensure isFlank is always true
							isFlank	 = true;
							currK    = sK[0];
							currFlank= flankLength;
							currKs   = new int[]{currK};
							nucSym   = null;
							if (hasSymmetry && useSymmetry[0]) {
								nucSym = Array.cat(nucSym, evenOdd(currK));
							} else {
								nucSym = Array.cat(nucSym, "null");
							}
							
							//Fit shifts and find best model for the first mode
							multiModel = new MultiModeModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), isFlank, currFlank, 
									false, false, isNSBinding, currKs, nucSym, null);
							multiModel.setLambda(lambda*multiModel.getNCount());
							currFits = optimizeModel(multiModel, datasets, null, true);
							latestFit = minFit(currFits);
							multiModel.threadPoolShutdown();

							//Now add modes
							for (int currMode=1; currMode<nModes; currMode++) {
								//Build seed & update model
								pos = latestFit.finalPosition;
								seed = (isNSBinding) ? Arrays.copyOfRange(pos, 0, pos.length-1) : Arrays.copyOfRange(pos, 0, pos.length);
								seed = Array.cat(seed, new double[4*sK[currMode]]);
								seed = (isNSBinding) ? Array.cat(seed, pos[pos.length-1]) : seed;
								savePrint(latestFit, true, results, false);
								currKs = Array.cat(currKs, sK[currMode]);
								if (hasSymmetry && useSymmetry[currMode]) {
									nucSym = Array.cat(nucSym, evenOdd(sK[currMode]));
								} else {
									nucSym = Array.cat(nucSym, "null");
								}
								//Set active mode to the newest one
								activeModes = new boolean[currMode+1];
								for (int i=0; i<currMode; i++) {
									activeModes[i] = false;
								}
								activeModes[currMode] = true;
								//Fit model for the new mode								
								multiModel = new MultiModeModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), isFlank, currFlank, 
										false, false, isNSBinding, currKs, nucSym, null);
								multiModel.setLambda(lambda*multiModel.getNCount());
								multiModel.setParams(seed);
								multiModel.setActiveModes(activeModes);
								currFits = optimizeModel(multiModel, datasets, seed, true);							
								latestFit = minFit(currFits);
								seed = latestFit.finalPosition;
								
								//Activate all modes & refit
								for (int i=0; i<currMode+1; i++) {
									activeModes[i] = true;
								}
								multiModel.setActiveModes(activeModes);
								currFits = optimizeModel(multiModel, seed);
								latestFit = minFit(currFits);
								latestFit.finalPosition = multiModel.normalize();
								multiModel.threadPoolShutdown();
							}
							savePrint(latestFit, true, results, false);
							System.out.println("Finished performing sequential fitting for starting k.");

							//All modes have been added; check to see if all modes have hit their
							//largest length. If not increase their length
							while (true) {
								hitMax = true;
								nucSym = null;
								seed   = null;
								for (int i=0; i<nModes; i++) {
									if (currKs[i] < mK[i]) {
										hitMax = false;
										currKs[i] += 2;
										seed = Array.cat(seed, Array.cat(Array.cat(new double[4], multiModel.getNucBetas(i)), new double[4]));
									} else {
										seed = Array.cat(seed, multiModel.getNucBetas(i));
									}
									//Build symmetry string
									if (hasSymmetry && useSymmetry[i]) {
										nucSym = Array.cat(nucSym, evenOdd(currKs[i]));
									} else {
										nucSym = Array.cat(nucSym, "null");
									}
								}
								if (isNSBinding) {
									seed = Array.cat(seed, multiModel.getNSBeta());
								}
								//Check to see if all modes have hit max length
								if (hitMax) {
									break;
								}
								//Grow flank length and check to see if it is within bounds
								currFlank++;
								if (l+2*currFlank-Array.max(currKs)+1<1 || l+2*currFlank> 32 || (isShape && l+2*currFlank+4 > 32)) {
									break;	
								}
								//Refit new model
								System.out.println("Fitting 2 additional positions from previous motif(s).");
								multiModel = new MultiModeModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), isFlank, currFlank, 
										false, false, isNSBinding, currKs, nucSym, null);
								multiModel.setLambda(lambda*multiModel.getNCount());
								currFits = optimizeModel(multiModel, seed);
								latestFit = minFit(currFits);
								latestFit.finalPosition = multiModel.normalize();
								savePrint(latestFit, true, results, false);
								multiModel.threadPoolShutdown();
							}
							//All modes extended to full length.
							savePrint(null, false, results, true);
							System.out.println("Finished growing motif(s).");

							//Add dinucleotides or shape features if requested
							if (useDinuc || isShape) {
								//Ensure normalized start point
								multiModel.setParams(multiModel.normalize());
								
								//Next, create the seed vector by adding dinucleotide or
								//shape parameters to nuc and define the symmetry structure 
								seed	= null;
								dinucSym= null;
								for (int currMode=0; currMode<nModes; currMode++) {
									seed = Array.cat(seed, multiModel.getNucBetas(currMode));
									seed = (useDinuc) ? Array.cat(seed, new double[16*(currKs[currMode]-1)]) : seed;
									seed = (isShape) ? Array.cat(seed, new double[shapeModel.nShapeFeatures()*currKs[currMode]]) : seed;
									if (hasSymmetry && useSymmetry[currMode] && useDinuc) {
										dinucSym= Array.cat(dinucSym, evenOdd(currKs[currMode]-1));
									} else {
										dinucSym= Array.cat(dinucSym, "null");
									}
								}
								if (isNSBinding) {
									seed = Array.cat(seed, multiModel.getNSBeta());
								}
								//Next, create new model and optimize it
								multiModel = new MultiModeModel(nThreads, shapeModel, ((Data) datasets.get(0)[0]), isFlank, 
										currFlank, useDinuc, isShape, isNSBinding, currKs, nucSym, dinucSym);
								multiModel.setLambda(lambda*multiModel.getNCount());
								System.out.println("Perfoming single-shot addition of dinucleotides and/or shape features.");
								currFits = optimizeModel(multiModel, seed);
								latestFit = minFit(currFits);
								latestFit.finalPosition = multiModel.normalize();
								savePrint(latestFit, true, results, true);
								multiModel.threadPoolShutdown();
							}
						}
					}
				}
			}
		}
		results.print(outputLocation+"/"+outputDataName, true, false, false);
		System.out.println("Process complete");
	}

	private static Fit[] optimizeModel(Model model, double[] seed) {
		return optimizeModel(model, null, seed, false, null);
	}

	private static Fit[] optimizeModel(Model model, ArrayList<Object[]> data,
			double[] seed, boolean testShifts) {
		return optimizeModel(model, data, seed, testShifts, null);
	}
	
	private static Fit[] optimizeModel(Model model, ArrayList<Object[]> data, 
			double[] seed, boolean testShifts, MultinomialResults results) {
		Minimizer min;
		Fit[] out;
		
		//Initialize minimizer
		if (minimizerType.equals("LBFGS")) {
			min = new LBFGS(model, lbfgsMem, lbfgsConvergence, lbfgsMaxIters, 
					lbfgsMCSearch, errorBars, storeHessian, isVerbose);
		} else {
			min = new PatternSearch(model, psInitStep, psConvergence, psTheta, 
					psRandomAxis, errorBars, storeHessian, isVerbose);
		}
		
		//Minimize model
		try {
			if (testShifts) {
				out = min.shiftPermutation(seed, null, nShifts, data, results);
			} else {
				out = min.minimize(seed, null);
				if (results!=null) {
					results.addFit(out[0]);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();			
			if (model instanceof MultinomialModel) {
				((MultinomialModel) model).threadPoolShutdown();				
			} else {
				((MultiModeModel) model).threadPoolShutdown();
			}
			return null;
		}
		return out;
	}
	
	private static String evenOdd(int k) {
		String output = "1";
		
		if (k%2==0) {
			for (int i=1; i<k/2; i++) {
				output = output.concat(",1");
			}
			for (int i=0; i<k/2; i++) {
				output = output.concat(",-1");
			}
		} else {
			for (int i=1; i<(k-1)/2; i++) {
				output = output.concat(",1");
			}
			output = output.concat(",*");
			for (int i=0; i<(k-1)/2; i++) {
				output = output.concat(",-1");
			}
		}
		return output;
	}
	
	
	private static MultinomialFit minFit(Fit[] input) {
		int currIdx = 0;
		double minVal = Double.MAX_VALUE;
		
		for (int i=0; i<input.length; i++) {
			if (input[i]!=null && input[i].trainLikelihood<minVal) {
				minVal = input[i].trainLikelihood;
				currIdx = i;
			}
		}
		return ((MultinomialFit) input[currIdx]);
	}
	
	private static void savePrint(MultinomialFit fit, boolean printFit, 
			MultinomialResults results, boolean isSave) {
		if (fit!=null) {
			results.addFit(fit);
			if (printFit) {
				System.out.println("-----Results for "+fit.betaIdx.length+" mode(s):");
				fit.print(true, false, false);
			}
		}
		if (isSave) {
			results.printList(outputLocation+"/"+outputDataName, false);
			results.printList(outputLocation+"/"+outputDataName, true);
		}
	}
	
	public static void readConfigFile(String configFile) throws Exception {
		boolean isHit;
		int nLengths;
		String currLine;
		int[] sK, mK, tempK;
		String[] temp, temp2;
		BufferedReader br	= new BufferedReader(new FileReader(configFile));
		config				= new ArrayList<String>(100);
		
		//First load all non-commented lines
		while ((currLine = br.readLine())!=null) {
			if (currLine.startsWith("#")) {
				continue;
			}
			config.add(currLine);
		}
		br.close();
		
		//Fixed values
		crossValidate		= false;
		testRatio			= .1;
		nFolds				= 1;
		isR0Flank			= true;
		printPSAM			= true;
		printSeeds			= false;
		printRaw			= false;
		
		//Read in required values: data file info and output info
		nThreads			= parseInt("nThreads", true);
		isVerbose			= parseBoolean("isVerbose", true);
		
		//Data info
		filterReads			= parseBoolean("filterReads", false);
		if (filterReads) {
			maxBaseRepeat	= parseInt("maxBaseRepeat", true);
			maxBaseCount	= parseInt("maxBaseCount", true);
			regexFilter		= extractArray("regexFilter", ",", true);
		}
		
		useDinuc			= parseBoolean("useDinuc", true);
		if (useDinuc) {
			growDinuc		= true;
		}
		shapes				= new ArrayList<String[]>();
		temp				= extractArray("useShape", ",", true);
		for (int i=0; i<temp.length; i++) {
			if (Boolean.parseBoolean(temp[i])) {	//consider shape features
				shapeDir	= parse("shapeDir", true);
				temp2		= extractArray("shapes", "\\|", true);
				for (int j=0; j<temp2.length; j++) {
					shapes.add(temp2[j].split(","));
				}				
			} else {								//do NOT consider shape features
				shapes.add(new String[]{"0"});
			}
		}
		useNSBinding		= extractArrayBoolean("useNSBinding", ",", true);
		nShifts				= parseInt("nShifts", true);
		testShifts			= (nShifts==0) ? false : true;
		flankLengths		= extractArrayInt("flankLength", ",", true);
		
		nModes				= parseInt("nModes", true);
		smSweep				= false;
		sK					= extractArrayInt("startK", ",", true);
		//Define fitting operations: first check if fits are seeded
		if (parseBoolean("seedFit", true)) {
			//Fits are seeded
			startK		= new ArrayList<int[]>();
			modeSeeds	= new ArrayList<double[]>();
			startK.add(sK);
			for (int i=1; i<nModes+1; i++) {
				modeSeeds.add(extractArrayDouble("modeSeed"+i,",", true));
			}
			isSeeded	= true;
			//Force nsbinding to be null
			nsBindingSeed	= null;
		} else {
			mK				= extractArrayInt("maxK", ",", true);
			//Fits are not seeded. Check to see if sweepLengths is required
			if (parseBoolean("sweepLengths", true)) {
				nLengths 	= mK[0]-sK[0]+1;	//number of lengths to sweep
				//Set smSweep if only one mode, and create appropiate testKs
				if (nModes==1) {
					smSweep	= true;
					testKs	= new int[nLengths];
					for (int iter=0; iter<nLengths; iter++) {
						testKs[iter] = sK[0]+iter;
					}
				} else {
					//For multiple modes, all modes have the same length
					startK	= new ArrayList<int[]>();
					maxK	= new ArrayList<int[]>();
					for (int iter=0; iter<nLengths; iter++) {
						//Create a temp array with identical values
						tempK	= new int[nModes];
						for (int iters=0; iters<nModes; iters++) {
							tempK[iters] = sK[0]+iter;
						}
						//Add it to the starK-maxK arrays
						startK.add(tempK);
						maxK.add(tempK);
					}
				}
			} else {
				//No need to sweep lengths. Read in startK-maxK normally
				if (sK.length!=nModes) throw new IllegalArgumentException("startK is incorrectly set!");
				if (mK.length!=nModes) throw new IllegalArgumentException("maxK is incorrectly set!");
				startK		= new ArrayList<int[]>();
				maxK		= new ArrayList<int[]>();
				startK.add(sK);
				maxK.add(mK);
			}
		}
		
		//Define symmetry status for the chosen number of modes
		useSymmetry			= extractArrayBoolean("useSymmetry", ",", true);
		if (!parseBoolean("seedFit", true) && parseBoolean("sweepLengths", true) && nModes!=1) {
			isHit = useSymmetry[0];
			useSymmetry = new boolean[nModes];
			for (int currMode=0; currMode<nModes; currMode++) {
				useSymmetry[currMode] = isHit;
			}
		} else if (useSymmetry.length!=nModes) {
			throw new IllegalArgumentException("useSymmetry is incorrectly set!");
		}
		hasSymmetry			= false;
		for (boolean currVal : useSymmetry) {
			if (currVal) {
				hasSymmetry = true;
				break;
			}
		}
		
		storeHessian		= parseBoolean("errorBars", true);
		errorBars			= parseBoolean("errorBars", true);
		lambda				= parseDouble("lambda", false);
		
		//Minimizer Configuration
		minimizerType		= parse("minimizerType", true);
		if (minimizerType.matches("(?i)lbfgs")) {			//case insensitive matching
			minimizerType	= "LBFGS";
			lbfgsMem		= parseInt("lbfgsMem", true);
			lbfgsMaxIters	= parseInt("lbfgsMaxIters", true);
			lbfgsConvergence= parseDouble("lbfgsConvergence", true);
			lbfgsMCSearch	= parseBoolean("lbfgsMCSearch", true);
		}
		if (minimizerType.matches("(?i)ps|(?i)patternsearch|(?i)pattern search")) {
			minimizerType	= "Pattern Search";
			psInitStep		= parseDouble("psInitStep", true);
			psTheta			= parseDouble("psTheta", true);
			psConvergence	= parseDouble("psConvergence", true);
			psRandomAxis	= parseBoolean("psRandomAxis", true);
		}
			
		sequencingRunName	= parse("sequencingRunName", true);
		sampleName			= parse("sampleName", true);
		l					= parseInt("varLen", true);
		samplePath			= parse("workingDir", true)+"/"+sequencingRunName+"."+sampleName+"."+l+".dat";
		lFlank				= parse("lFlank", true);
		rFlank				= parse("rFlank", true);
		R0ModelPath			= parse("R0ModelPath", true);
		outputLocation		= parse("outputPath", true);
		outputDataName		= parse("outputName", false);		
	}
	
	public static boolean parseBoolean(String paramName, boolean isRequired) {
		return Boolean.parseBoolean(parse(paramName, isRequired));
	}
	
	public static int parseInt(String paramName, boolean isRequired) {
		return Integer.parseInt(parse(paramName, isRequired));
	}
	
	public static double parseDouble(String paramName, boolean isRequired) {
		return Double.parseDouble(parse(paramName, isRequired));
	}
	
	public static String parse(String paramName, boolean isRequired) {
		String output = null, currLine;
		String[] parsed;
		
		for (int i=0; i<config.size(); i++) {
			currLine= config.get(i).replaceAll("\"", "");					//remove all quotes if they exist
			currLine= config.get(i).replaceAll("\'", "");
			parsed	= currLine.split("=");									//split the first and second halves of the entry
			if ((parsed[0].trim()).matches("(?i)"+paramName)) {				//Perform case-insensitive matching
				output = (currLine.split("=")[1]).split("#|;")[0].trim();	//Remove trailing comments or semicolons
			}
		}
		if (output==null && isRequired) {
			throw new IllegalArgumentException("Configuration File Does Not Contain The Required Entry: "+paramName);
		}
		if (!isRequired) {
			if (output!=null && output.matches("(?i)null|(?i)na|(?i)n/a|(?i)false|(?i)no")) {
				output = null;
			}
		}
		
		return output;
	}
	
	
	public static String[] extractArray(String paramName, String split, boolean isRequired) {
		String parsed = parse(paramName, isRequired);
		
		if (parsed==null) {
			return null;
		} else {
			return parsed.replaceAll("\\s+", "").split(split);
		}
	}
	

	public static int[] extractArrayInt(String paramName, String split, boolean isRequired) {
		int[] output;
		String[] parsed = extractArray(paramName, split, isRequired);
		
		if (parsed==null) {
			return null;
		} else {
			output = new int[parsed.length];
			for (int i=0; i<parsed.length; i++) {
				output[i] = Integer.parseInt(parsed[i]);
			}
			return output;
		}
	}
	
	
	public static double[] extractArrayDouble(String paramName, String split, boolean isRequired) {
		double[] output;
		String[] parsed = extractArray(paramName, split, isRequired);
		
		if (parsed==null) {
			return null;
		} else {
			output = new double[parsed.length];
			for (int i=0; i<parsed.length; i++) {
				output[i] = Double.parseDouble(parsed[i]);
			}
			return output;
		}
	}
	
	
	public static boolean[] extractArrayBoolean(String paramName, String split, boolean isRequired) {
		boolean[] output;
		String[] parsed = extractArray(paramName, split, isRequired);
		
		if (parsed==null) {
			return null;
		} else {
			output = new boolean[parsed.length];
			for (int i=0; i<parsed.length; i++) {
				output[i] = Boolean.parseBoolean(parsed[i]);
			}
			return output;
		}
	}
}
