package utils;

import java.lang.reflect.Array;
import java.util.Arrays;

import main.SELEX;
import main.SimpleKmerCount;
import base.DebugLog;
import base.MarkovModelInfo;
import base.MarkovModelOption;
import config.ExperimentReference;

public class Build_MM_Kmer_Table {
	public static boolean countKMers	= false;
	public static boolean calcKMax		= false;
	public static int l			= 16;
	public static int countRound		= 0;
	public static int markovModelLength	= 6;
	public static int markovModelKMax	= 8;
	public static String workingDir 	= "F:/Data/Datasets/tmp/";
	public static String config_path	= "./config/Max-R0.xml";
	public static String seqRunName		= "Antp1";
	public static String countSample	= "initialPool.16mer4";
	public static String trainSample	= "initialPool.16mer3";
	public static String testSample		= "initialPool.16mer4";
	
	public static void main(String[] args) {
		//Parse command line arguments
		if (args.length>0) {
			if (args.length!=6) {
				throw new IllegalArgumentException("Incorrect number of arguments supplied!");
			}
			countKMers = true;	//only count kmers
			l = Integer.parseInt(args[0]);
			countRound = Integer.parseInt(args[1]);
			workingDir = args[2];
			config_path= args[3];
			seqRunName = args[4];
			countSample= args[5];
			//System.out.println(l+" "+countRound+" "+workingDir+" "+config_path+" "+seqRunName+" "+countSample);
		}
		SELEX.setWorkingDirectory(workingDir);
		SELEX.loadConfigFile(config_path);
		DebugLog.log(Arrays.toString(SELEX.showSamples()));
		
		if (countKMers) {
			testKmerCounts();	//Performs K-mer Counting
		} else {
			testMarkov();		//Builds Markov Model
		}
	}

	public static void testMarkov() {
		ExperimentReference training= SELEX.getExperimentReference(seqRunName, trainSample, 0);
		ExperimentReference testing	= SELEX.getExperimentReference(seqRunName, testSample, 0);		
		int kmax = (calcKMax) ? SELEX.kmax(testing) : markovModelKMax;
		
		MarkovModelInfo mm=SELEX.trainMarkovModel(training, testing, markovModelLength,  kmax, null, 
				MarkovModelOption.DIVISION);
		DebugLog.log(mm.getMarkovR2());
	}
	
	private static void testKmerCounts() {		
		ExperimentReference dataSet=  SELEX.getExperimentReference(seqRunName,countSample, countRound);
		
		SimpleKmerCount obj=new SimpleKmerCount();
		obj.initTraining(SELEX.getConfigReader(), dataSet);
		obj.setTempFolder("./tmp/");
		int offset = -1;
		int mincount = -1;
		int top = 100;
		SELEX.doMinimalCounting(dataSet, l,  obj, null, false, offset,false,-1, null);
		
		Object[] counts = SELEX.getKmerCount(dataSet, l, offset, mincount, top, null, null);
		DebugLog.log("=================COUNTS==No Top===============");
		int len=Array.getLength(counts[0]);

		for(int i=0;i<len;i++)
		{
			StringBuffer sb=new StringBuffer();
			for(Object ds:counts)
			{
				sb.append(Array.get(ds, i)+"\t");
			}
			DebugLog.log(sb.toString());
		}
	}	
}
