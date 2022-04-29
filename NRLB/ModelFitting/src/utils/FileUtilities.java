package utils;

import java.util.ArrayList;
import java.util.Random;
import java.util.regex.Pattern;

import base.*;
import model.*;

public class FileUtilities {
	public int l, removedReads;
	private boolean stringFilter, repeatFilter;
	private int maxBaseCount, maxBaseRepeat;
	private Pattern regex;
	
	public FileUtilities(int l) {
		this.l				= l;
	}
	
	public static Data mergeData(Data ... input) {
		boolean isR0;
		int l, totElems = 0, offset = 0, totCounts = 0;
		int[] counts	= null;
		long[] probes	= null;
		double[] r0		= null;
		Data output;
		
		//Make sure input Data's are compatible 
		if (input.length==1) {
			return input[0];
		}
		l	= input[0].l;
		isR0= (input[0].R0Prob!=null) ? true : false;
		for (int i=0; i<input.length; i++) {
			if (input[i].probes==null && input[i].counts==null) {
				throw new IllegalArgumentException("Input Data Object "+i+" is Incomplete!");
			}
			if (input[i].l!=l && (input[i].R0Prob!=null)!=isR0) {
				throw new IllegalArgumentException("Input Data Objects are Incompatible!");
			}
			totElems += input[i].probes.length;
			totCounts+= input[i].nCount;
		}
		
		//Merge
		if (isR0) {
			counts	= new int[totElems];
			probes	= new long[totElems];
			r0		= new double[totElems];
			for (int i=0; i<input.length; i++) {
				for (int j=0; j<input[i].probes.length; j++) {
					probes[offset+j]= input[i].probes[j];
					counts[offset+j]= input[i].counts[j];
					r0[offset+j]	= input[i].counts[j];
				}
				offset += input[i].probes.length;
			}
		} else {
			counts = new int[totElems];
			probes = new long[totElems];
			for (int i=0; i<input.length; i++) {
				for (int j=0; j<input[i].probes.length; j++) {
					probes[offset+j] = input[i].probes[j];
					counts[offset+j] = input[i].counts[j];
				}
				offset += input[i].probes.length;
			}
		}
		
		output			= input[0].clone();
		output.nCount	= totCounts;
		output.probes	= probes;
		output.counts	= counts;
		output.R0Prob	= r0;
		return output;
	}
	
	/*
	 * WILL NOT WORK FOR l = 32! 
	 */
	public static void combineData(Data[] input, long[] outputProbes, int[][] outputCounts) {
		boolean isZero	= true;
		int nSets		= input.length;
		int setsLeft	= nSets;
		int idx;
		long minValue;
		int[] bestIdx, countEntry;
		int[] currCounts= new int[nSets];
		int[] maxIdx	= new int[nSets];
		int[] currIdx	= new int[nSets];
		long[] currSeqs	= new long[nSets];
		ArrayList<long[]> probeSet		= new ArrayList<long[]>(nSets);
		ArrayList<int[]> countSet		= new ArrayList<int[]>(nSets);
		ArrayList<Long> combinedProbes	= new ArrayList<Long>((int)1E7*nSets);
		ArrayList<int[]> combinedCounts	= new ArrayList<int[]>((int)1E7*nSets);
		
		//First build count sets and initialize maxIdx, currIdx
		for (int currSet=0; currSet<nSets; currSet++) {
			probeSet.add(input[currSet].probes);
			countSet.add(input[currSet].counts);
			maxIdx[currSet]		= input[currSet].probes.length;
			currIdx[currSet]	= 0;
			currSeqs[currSet]	= probeSet.get(currSet)[0];
			currCounts[currSet]	= countSet.get(currSet)[0];
		}
		
		//Loop until all reads for all sets are accounted for
		while (setsLeft>0) {
			//Find the smallest probe sequence(s)
			minValue	= currSeqs[0];
			bestIdx		= new int[]{0};
			for (int i=1; i<nSets; i++) {
				if (currSeqs[i] < minValue) {
					minValue= currSeqs[i];
					bestIdx	= new int[]{i};
				} else if (currSeqs[i]==minValue) {
					bestIdx = Array.cat(bestIdx, i);
				}
			}
			
			//Grab count values, create combined Array and get new probe values
			if (bestIdx.length==1) {
				idx = bestIdx[0];
				combinedProbes.add(currSeqs[idx]);
				countEntry		= new int[nSets];
				countEntry[idx]	= currCounts[idx];
				combinedCounts.add(countEntry);
				currIdx[idx]++;
				if (currIdx[idx]>=maxIdx[idx]) {
					currSeqs[idx]	= Long.MAX_VALUE;
					currCounts[idx] = 0;
					setsLeft--;
				} else {
					currSeqs[idx]	= probeSet.get(idx)[currIdx[idx]];
					currCounts[idx]	= countSet.get(idx)[currIdx[idx]];
				}
			} else {
				combinedProbes.add(currSeqs[bestIdx[0]]);
				countEntry		= new int[nSets];
				for (int cIdx : bestIdx) {
					countEntry[cIdx]	= currCounts[cIdx];
					currIdx[cIdx]++;
					if (currIdx[cIdx]>=maxIdx[cIdx]) {
						currSeqs[cIdx]	= Long.MAX_VALUE;
						currCounts[cIdx]= 0;
						setsLeft--;
					} else {
						currSeqs[cIdx]	= probeSet.get(cIdx)[currIdx[cIdx]];
						currCounts[cIdx]= countSet.get(cIdx)[currIdx[cIdx]];
					}
				}
				combinedCounts.add(countEntry);
			}
		}
		//See if the last entry (polyT) has non-zero entries. If not, delete
		idx = combinedCounts.size()-1;
		if (combinedProbes.get(idx)==Long.MAX_VALUE) {
			countEntry = combinedCounts.get(idx);
			for (int cIdx : countEntry) {
				if (cIdx!=0)	isZero = false;
			}
			if (isZero) {
				combinedProbes.remove(idx);
				combinedCounts.remove(idx);
			}
		}
		//Convert to Arrays
		outputProbes = new long[idx+1];
		outputCounts = new int[idx+1][nSets];
		for (int i=0; i<idx+1; i++) {
			outputProbes[i] = combinedProbes.get(i);
			outputCounts[i] = combinedCounts.get(i);
		}
		return;
	}
	
	
	public Data[] readSeqFile(String samplePath, String lFlank, String rFlank, 
			boolean isSplit) {
		Data[] output			= null;
		ArrayList<Object[]> out = readSeqFile(samplePath, null, lFlank, 
				rFlank, isSplit, 1, .5, false, 0, 0, null);
		
		if (isSplit) {
			output		= new Data[3];
			output[0]	= (Data) out.get(0)[0];
			output[1]	= (Data) out.get(1)[0];
			output[2]	= (Data) out.get(1)[1];
		} else {
			output = new Data[]{(Data) out.get(0)[0]};
		}
		return output;
	}
	
	public Data readSeqFile(String samplePath, boolean repeatFilter,
			int maxBaseCount, int maxBaseRepeat, String[] regexFilter) {
		ArrayList<Object[]> output = readSeqFile(samplePath, null, null, null, 
				false, 0, 0, repeatFilter, maxBaseCount, maxBaseRepeat, 
				regexFilter);
		return ((Data) output.get(0)[0]);
	}
	
	public ArrayList<Object[]> readSeqFile(String samplePath,
			Round0Model R0Model, String lFlank, String rFlank, 
			boolean crossValidate, int nFolds, double testRatio, 
			boolean repeatFilter, int maxBaseCount, int maxBaseRepeat, 
			String[] regexFilter) {
		boolean isR0				= (R0Model!=null) ? true : false;
		int currCount, totCount = 0;
		long currSeq;
		String regexString;
		ArrayList<Long> probeList	= new ArrayList<Long>(10*1000*1000);
		long[] probes;
		ArrayList<Integer> countList= new ArrayList<Integer>(10*1000*1000);
		int[] counts;
		ArrayList<Double> probList	= (isR0) ? new ArrayList<Double>(10*1000*1000) : null;
		double[] probs				= null;
		Random generator			= new Random();
		ArrayList<Object[]> output	= new ArrayList<Object[]>();
		CountObject obj;
		Data[] currDataObjs;
		ArraysMergerHeap.MinHeapNode node = new ArraysMergerHeap.MinHeapNode(samplePath);
		
		this.repeatFilter	= repeatFilter;
		this.maxBaseCount	= maxBaseCount;
		this.maxBaseRepeat	= maxBaseRepeat;
		if (regexFilter!=null) {
			stringFilter	= true;
			regexString		= regexFilter[0];
			for (int i=1; i<regexFilter.length; i++) {
				regexString	+= "|"+regexFilter[i];
			}
			regex			= Pattern.compile(regexString);
		} else {
			stringFilter	= false;
		}
		
		//First extract all probes from file and store
		currDataObjs = new Data[1];
		if (isR0) {
			if (repeatFilter || stringFilter) {
				removedReads = 0;
				while ((obj = node.peek()) != null) {
					currSeq 	= obj.getKey().getValue();
					currCount	= obj.getCount();
					if (!filter(currSeq)) {
						totCount	+= currCount;
						probeList.add(currSeq);
						countList.add(currCount); 
						probList.add(R0Model.getSeqProb(currSeq));
					}
					removedReads += currCount;
					node.pop();
				}
				removedReads -= totCount;
			} else {
				while ((obj = node.peek()) != null) {
					currSeq 	= obj.getKey().getValue();
					currCount	= obj.getCount(); 
					totCount	+= currCount;
					probeList.add(currSeq);
					countList.add(currCount); 
					probList.add(R0Model.getSeqProb(currSeq));
					node.pop();
				}
			}
			probes = new long[probeList.size()];
			counts = new int[probeList.size()];
			probs  = new double[probeList.size()];
			for (int i=0; i<probeList.size(); i++) {
				probes[i] = probeList.get(i);
				counts[i] = countList.get(i);
				probs[i]  = probList.get(i);
			}
		} else {
			if (repeatFilter || stringFilter) {
				removedReads = 0;
				while ((obj = node.peek()) != null) {
					currSeq 	= obj.getKey().getValue();
					currCount	= obj.getCount();
					if (!filter(currSeq)) {
						totCount	+= currCount;
						probeList.add(currSeq);
						countList.add(currCount); 
					}
					removedReads += currCount;
					node.pop();
				}
				removedReads -= totCount;
			} else {
				while ((obj = node.peek()) != null) {
					currSeq 	= obj.getKey().getValue();
					currCount	= obj.getCount(); 
					totCount	+= currCount;
					probeList.add(currSeq);
					countList.add(currCount); 
					node.pop();
				}
			}
			probes = new long[probeList.size()];
			counts = new int[probeList.size()];
			for (int i=0; i<probeList.size(); i++) {
				probes[i] = probeList.get(i);
				counts[i] = countList.get(i);
			}
		}
		currDataObjs[0] = new Data(l, totCount, lFlank, rFlank, counts, probes, probs, R0Model);
		output.add(currDataObjs);
		System.out.println("Finished reading data.");
		
		//Now construct cross-validation set(s)
		if(crossValidate) {
			int testCount, trainSet, testSet;
			ArrayList<Long> testProbes, trainProbes;
			ArrayList<Integer> testCounts, trainCounts;
			ArrayList<Double> testProb, trainProb;

			for (int i=0; i<nFolds; i++) {			//Loop over the number of folds
				totCount	= 0;
				testCount	= 0;
				trainProbes	= new ArrayList<Long>(10*1000*1000);
				trainCounts	= new ArrayList<Integer>(10*1000*1000);
				trainProb	= (isR0) ? new ArrayList<Double>(10*1000*1000) : null;
				testProbes	= new ArrayList<Long>(10*1000*1000);
				testCounts	= new ArrayList<Integer>(10*1000*1000);
				testProb	= (isR0) ? new ArrayList<Double>(10*1000*1000) : null;
				currDataObjs= new Data[2];
				
				if (isR0) {
					for (int j=0; j<probeList.size(); j++) {
						trainSet	= 0;
						testSet		= 0;
						for (int nReads=0; nReads<countList.get(j); nReads++) {
							if (generator.nextDouble() < testRatio) {		//Stash a percentage of reads away as the test set
								testSet++;
							} else {
								trainSet++;
							}
						}
						if (testSet!=0) {
							testCount += testSet;
							testProbes.add(probeList.get(j));
							testCounts.add(testSet);
							testProb.add(probList.get(j));
						}
						if (trainSet!=0) {
							totCount += trainSet;
							trainProbes.add(probeList.get(j));
							trainCounts.add(trainSet);
							trainProb.add(probList.get(j));
						}
					}					
				} else {
					for (int j=0; j<probeList.size(); j++) {
						trainSet	= 0;
						testSet		= 0;
						for (int nReads=0; nReads<countList.get(j); nReads++) {
							if (generator.nextDouble() < testRatio) {		//Stash a percentage of reads away as the test set
								testSet++;
							} else {
								trainSet++;
							}
						}
						if (testSet!=0) {
							testCount += testSet;
							testProbes.add(probeList.get(j));
							testCounts.add(testSet);
						}
						if (trainSet!=0) {
							totCount += trainSet;
							trainProbes.add(probeList.get(j));
							trainCounts.add(trainSet);
						}
					}
				}

				//store train data
				probes = new long[trainProbes.size()];
				counts = new int[trainProbes.size()];
				probs  = (isR0) ? new double[trainProbes.size()] : null;
				if (isR0) {
					for (int k=0; k<trainProbes.size(); k++) {
						probes[k] = trainProbes.get(k);
						counts[k] = trainCounts.get(k);
						probs[k]  = trainProb.get(k);
					}	
				} else {
					for (int k=0; k<trainProbes.size(); k++) {
						probes[k] = trainProbes.get(k);
						counts[k] = trainCounts.get(k);
					}
				}
				currDataObjs[0] = new Data(l, totCount, lFlank, rFlank, counts, probes, probs, R0Model);
				//store test data
				probes = new long[testProbes.size()];
				counts = new int[testProbes.size()];
				probs  = (isR0) ? new double[testProbes.size()] : null;
				if (isR0) {
					for (int k=0; k<testProbes.size(); k++) {
						probes[k] = testProbes.get(k);
						counts[k] = testCounts.get(k);
						probs[k]  = testProb.get(k);
					}	
				} else {
					for (int k=0; k<testProbes.size(); k++) {
						probes[k] = testProbes.get(k);
						counts[k] = testCounts.get(k);
					}
				}
				currDataObjs[1] = new Data(l, testCount, lFlank, rFlank, counts, probes, probs, R0Model);
				output.add(currDataObjs);
			}
			System.out.println("Finished constructing cross-validation datasets.");
		}
		return output;
	}
	
	private boolean filter(long seq) {
		if (repeatFilter) {
			if (stringFilter) {
				return (repeatFilter(seq) || 
					regex.matcher((new Sequence(seq, l).getString())).find());
			} else {
				return repeatFilter(seq);
			}
		} else {
			return regex.matcher((new Sequence(seq, l).getString())).find();
		} 
	}
		
	private boolean repeatFilter(long seq) {
		boolean isBase;
		int maxLength, currLength;
		long currChar, currSeq = seq;
		int[] baseCounts = new int[4];

		//Filter steps below
		for (int i=0; i<l; i++) {
			currChar= (currSeq&3);
			baseCounts[(int) currChar]++;
			currSeq	>>= 2;
		}
		for (int j=0; j<4; j++) {
			if (baseCounts[j] >= maxBaseCount) {
				currSeq = seq;
				currLength = 0;
				maxLength = 0;
				isBase = false;
				for (int i=0; i<l; i++) {
					currChar = (currSeq&3);
					if (((int) currChar)==j) {
						if (isBase) {
							currLength++;
						} else {
							isBase = true;
							currLength = 1;
						}
					} else {
						isBase = false;
						if (currLength > maxLength) {
							maxLength = currLength;
						}
						currLength = 0;
					}
					currSeq >>= 2;
				}
				if (currLength > maxLength) {
					maxLength = currLength;
				}
				if (maxLength >= maxBaseRepeat) {
					return true;
				}
			}
		}
		return false;
	}
	
	protected static long rc(long input, int length) {
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
}