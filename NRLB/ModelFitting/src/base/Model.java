package base;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public abstract class Model {
	public int nullVectors;
	public int maxFitRetries	= 5;
	public double evCutoff		= 1E-12;
	public Matrix[] M			= null;
	protected double[] errorBars;
	protected double[][] hessian;
	
	//Va
	private Map<Integer, Boolean> symmetryCMap = new HashMap<>(); //Is the base-feature complement-symmetric?
	private ArrayList<Integer> symmetryIDList = new ArrayList<Integer>(); //IDs of bases along binding site. Negative is complement.
	private ArrayList<Boolean> symmetryBarrierList = new ArrayList<Boolean>(); //Indicators of barriers across binding site. 
	
	
	//The user must check to see if o is the right data type
	public abstract void replaceData(Object o);
	
	public abstract double likelihoodNormalizer();
	
	public abstract double maxLikelihood();
	
	public abstract void normalForm();
	
	//Return the number of FULL dimensions (INCLUDES symmetrized dimensions)
	public abstract int getTotFeatures();
	
	//Return the number of TRUE dimensions (the dimensionality of the symmetry-reduced space)
	public abstract int getNDimensions();

	//setParams does NOT SYMMETRIZE input
	public abstract void setParams(double[] position);
	
	public abstract double[] getPositionVector();
	
	public abstract double[] shiftBetas(double[] originalBetas, int shiftPositions);
	
	public abstract double[] orthogonalStep(double[] currPos, int position, double stepSize);

	//None of the following 5 functions utilize or obey symmetries. 
	public abstract double functionEval() throws Exception;
	
	public abstract CompactGradientOutput gradientEval() throws Exception;
	
	public abstract CompactGradientOutput getGradient();
	
	public abstract CompactGradientOutput hessianEval() throws Exception;
	
	public CompactGradientOutput getHessian() {
		if (hessian==null) {
			return null;
		}
		return new CompactGradientOutput(hessian);
	}
		
	public double[] errorEval() {
		boolean isNaN				= true;
		int nDims					= this.getNDimensions();
		double tol					= evCutoff;
		double currError;
		errorBars 					= new double[getTotFeatures()];
		double[] absEV				= new double[nDims];
		double[] realEV, complexEV;
		double[][] hessian, inv;
		double[][] DMatrix			= new double[nDims][nDims];
		Matrix V, D;
		EigenvalueDecomposition ev;
		CompactGradientOutput output= null;
		
		//compute hessian
		try {
			output = hessianEval();
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (output==null || output.hessian==null) {			//handle the case where the hessian is undefined
			return null;
		}
		
		//Moore-Penrose Psuedoinverse
		//Compute SVD on REDUCED hessian
		hessian = Array.symmetrize(output.hessian);
		hessian = this.compressHessian(hessian);
		ev = new EigenvalueDecomposition(new Matrix(hessian));
		//Get singular values
		realEV		= ev.getRealEigenvalues();
		complexEV	= ev.getImagEigenvalues();
		for (int j=0; j<nDims; j++) {
			absEV[j] = Math.sqrt(realEV[j]*realEV[j]+complexEV[j]*complexEV[j]);
		}
		absEV = Array.maxNormalize(absEV);
		
		while (isNaN) {
			V = ev.getV().copy();
			D = ev.getD().copy();
			//Remove singular eigenvectors and eigenvalues
			for (int j=0; j<nDims; j++) {
				if (Math.abs(absEV[j])<tol) {		
					DMatrix[j][j] = 0;
				} else {
					DMatrix[j][j] = 1/D.get(j,j);
				}
			}
			D = new Matrix(DMatrix);
			//invert hessian
			inv = this.uncompressHessian(V.times(D.times(V.transpose())).getArray());
			for (int j=0; j<getTotFeatures(); j++) {
				currError = Math.sqrt(inv[j][j]);
				if (Double.isNaN(currError)) {
					tol = tol*10;
					isNaN = true;
					break;
				}
				errorBars[j] = currError;
				isNaN = false;
			}
		}
		nullVectors = 0;
		for (double currEV : absEV) {
			if (currEV<tol)		nullVectors++;
		}
		return Array.clone(errorBars);
	}
	
	public double[] getErrorBars() {
		return Array.clone(errorBars);
	}
	
	public abstract Fit generateFit(double[] seed);
	
	protected long reverse(long input, int length) {
		long output = 0;
		
		for (int i=0; i<length; i++) {
			output = ((output << 2) | (input & 3));
			input = input >> 2;
		}
		return output;
	}
	
	protected long reverseComplement(long input, int length) {
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
	
	protected ArrayList<int[]> parseSymmetryString(String symString) {
		int delim				= 111;
		int selfComplementDelim	= 222;
		int totSum				= 0;
		int commaPos, pipePos, currGroup, idx, currIdx, nElems;
		boolean[] isHit			= null;
		int[] structure			= null;
		int[] currFwdStructure, currRevStructure, tempArray;
		ArrayList<int[]> group	= new ArrayList<int[]>();
		ArrayList<int[]> linked	= new ArrayList<int[]>();
		
		symString = symString.trim();
		while (symString.indexOf(",")>0 || symString.indexOf("|")>0) {	//comma OR pipe exist
			commaPos= symString.indexOf(",");
			pipePos	= symString.indexOf("|");
			if (commaPos>0) {		//comma exists, pipe may or may not exist
				if (pipePos>0) {	//Pipe exists
					if (commaPos<pipePos) {		//comma exists before pipe
						if (symString.charAt(0)=='*') {
							structure	= Array.cat(structure, new int[]{selfComplementDelim});
						} else {
							currGroup	= Integer.parseInt(symString.substring(0,commaPos));
							structure	= Array.cat(structure, new int[]{currGroup});
						}
						symString	= symString.substring(commaPos+1);
					} else {					//pipe exists before comma
						if (symString.charAt(0)=='*') {
							structure	= Array.cat(structure, new int[]{selfComplementDelim});
						} else {
							currGroup	= Integer.parseInt(symString.substring(0,pipePos));
							structure	= Array.cat(structure, new int[]{currGroup, delim});
						}
						symString	= symString.substring(pipePos+1);
					}
				} else {			//comma exists
					if (symString.charAt(0)=='*') {
						structure	= Array.cat(structure, new int[]{selfComplementDelim});
					} else {
						currGroup	= Integer.parseInt(symString.substring(0,commaPos));
						structure	= Array.cat(structure, new int[]{currGroup});
					}
					symString	= symString.substring(commaPos+1);
				}
			} else {				//comma does NOT exist therefore pipe exists
				if (symString.charAt(0)=='*') {
					structure	= Array.cat(structure, new int[]{selfComplementDelim});
				} else {
					currGroup	= Integer.parseInt(symString.substring(0,pipePos));
					structure	= Array.cat(structure, new int[]{currGroup, delim});
				}
				symString	= symString.substring(pipePos+1);
			}
			symString = symString.trim();
		}
		currGroup	= Integer.parseInt(symString);
		structure	= Array.cat(structure, new int[]{currGroup});
		
		//Test to see if it is a valid string
		isHit		= new boolean[structure.length];
		for (int i=0; i<structure.length; i++) {
			totSum += (structure[i]==delim || structure[i]==selfComplementDelim) ? 0 : structure[i];
		}
		if (totSum!=0)	throw new IllegalArgumentException("Symmetry input string is incorrect");
		
		for (int i=0; i<structure.length; i++) {
			if (isHit[i])	continue;
			if (structure[i]==0) {
				idx = 0;
				for (int j=1; j<=i; j++) {
					if (structure[j]!=delim && structure[j]!=selfComplementDelim)	idx++;
				}
				linked.add(new int[]{idx});
				isHit[i] = true;
				continue;
			}
			if (structure[i]==selfComplementDelim) {
				idx = 0;
				for (int j=1; j<=i; j++) {
					if (structure[j]!=delim) 	idx++;
				}
				linked.add(new int[]{-idx});
				isHit[i] = true;
				continue;
			}
			//First find all the elements of the same group
			group			= new ArrayList<int[]>();
			currIdx			= 0;
			currGroup		= structure[i];
			currFwdStructure= null;
			currRevStructure= null;
			for (int j=0; j<structure.length; j++) {
				if (structure[j]==delim) {
					if (structure[j-1]==currGroup && structure[j+1]==currGroup) {
						currFwdStructure = Array.cat(currFwdStructure, new int[]{delim});
					} else if(structure[j-1]==-currGroup && structure[j+1]==-currGroup) {
						currRevStructure = Array.cat(new int[]{delim}, currRevStructure);
					}
					isHit[j] = true;
				} else {
					if (structure[j]==currGroup) {
						currFwdStructure = Array.cat(currFwdStructure, new int[]{currIdx});
					} else if(structure[j]==-currGroup) {
						currRevStructure = Array.cat(new int[]{-currIdx}, currRevStructure);
					}
					currIdx++;
				}
			}
			//Process subgroups
			tempArray = new int[]{currFwdStructure[0]};
			for (int j=1; j<currFwdStructure.length; j++){
				if (currFwdStructure[j]-currFwdStructure[j-1]>1) {
					if (currFwdStructure[j]==delim) {
						j++;
					}
					group.add(tempArray);
					tempArray = new int[]{currFwdStructure[j]};
				} else {
					tempArray = Array.cat(tempArray, new int[]{currFwdStructure[j]});
				}
			}
			group.add(tempArray);
			tempArray = new int[]{currRevStructure[0]};
			for (int j=1; j<currRevStructure.length; j++){
				if (currRevStructure[j-1]-currRevStructure[j]<-1 || currRevStructure[j]==delim) {
					if (currRevStructure[j]==delim) {
						j++;
					}
					group.add(tempArray);
					tempArray = new int[]{currRevStructure[j]};
				} else {
					tempArray = Array.cat(tempArray, new int[]{currRevStructure[j]});
				}
			}
			group.add(tempArray);
			//Ensure that every subgroup is of the same length
			nElems = group.get(0).length;
			for (int j=1; j<group.size(); j++) {
				if (group.get(j).length!=nElems){
					throw new IllegalArgumentException("Symmetry input string is incorrect");
				}
			}
			//Create a list of positions that are linked
			for (int j=0; j<nElems; j++) {
				tempArray = new int[group.size()];
				for (int k=0; k<group.size(); k++) {
					tempArray[k] = group.get(k)[j];
				}
				linked.add(tempArray);
			}
			//Mark positions as processed
			for (int j=0; j<structure.length; j++) {
				if (structure[j]==currGroup || structure[j]==-currGroup) {
					isHit[j] = true;
				}
			}
		}
		return linked;
	}
	
	protected Matrix[] matrixBuilder(ArrayList<double[]> matrix, int maxFeatures) {
		double[][] intM = new double[matrix.size()][maxFeatures];
		double[][] intMp= new double[matrix.size()][maxFeatures];
		
		for (int i=0; i<matrix.size(); i++) {
			intM[i] = matrix.get(i);
		}
		//Create degenerate matrix Mp
		for (int i=0; i<intM.length; i++) {
			for (int j=0; j<maxFeatures; j++) {
				if (intM[i][j]==1) {
					intMp[i][j] = 1;
					break;
				}
			}
		}
		
		return new Matrix[]{new Matrix(intM), new Matrix(intMp)};
	}
	
	protected Matrix[] mMatrix(ArrayList<int[]> structure, int blockSize, int nBases) {
		int currOffset, maxFeatures = 0;
		int[] currGroup;
		double[] work;
		ArrayList<double[]> matrix = new ArrayList<double[]>();
		
		//First find vector length
		for (int i=0; i<structure.size(); i++) {
			currGroup = structure.get(i);
			for (int j=0; j<currGroup.length; j++) {
				if (Math.abs(currGroup[j]) > maxFeatures) {
					maxFeatures = Math.abs(currGroup[j]);
				}
			}
		}
		maxFeatures = (maxFeatures+1)*blockSize;
		
		//Now compute vectors for each group
		for (int group=0; group<structure.size(); group++) {
			currGroup = structure.get(group);
			//Handle the case of no symmetry or self-symmetry
			if(currGroup.length==1) {
				currOffset = Math.abs(currGroup[0]);
				//No symmetry
				if (currGroup[0]>=0 || (currGroup[0]<0 && nBases==0)) {
					for (int i=0; i<blockSize; i++) {
						work = new double[maxFeatures];
						work[currOffset*blockSize+i] = 1;
						matrix.add(work);
					}
				} else {							//Self-symmetry
					for (int i=0; i<blockSize; i++) {
						if (reverseComplement(i, nBases) >= i) {
							work = new double[maxFeatures];
							work[currOffset*blockSize+i] = 1;
							work[currOffset*blockSize+(int) reverseComplement(i, nBases)] = 1;
							matrix.add(work);
						}
					}
				}
			} else {								//Nucleotide reverse complement symmetry
				for (int i=0; i<blockSize; i++) {	//Loop over positions within a block
					work = new double[maxFeatures];
					for (int j=0; j<currGroup.length; j++) {
						currOffset = Math.abs(currGroup[j]);
						//Ignore RC symmetry if not a nucleotide representation
						if (currGroup[j]>=0 || nBases==0) {
							work[currOffset*blockSize+i] = 1;
						} else {
							work[currOffset*blockSize+(int) reverseComplement(i, nBases)] = 1;
						}
					}
					matrix.add(work);
				}
			}
		}

		return matrixBuilder(matrix, maxFeatures);
	}
	
	public int nSymmetrizedFeatures() {
		if (M==null) {
			return 0;
		}
		return M[0].getArray().length;
	}
	
	public boolean isSymmetrized(int position) {
		if (M==null) {
			return false;
		}
		
		for (int i=0; i<M[0].getArray().length; i++) {
			if (M[0].getArray()[i][position]==1) {		//Found the right feature
				if (M[1].getArray()[i][position]==0) {	//Not the 'source feature'
					return true;
				} else {
					return false;
				}
			}
		}
		return false;
	}
	
	public double[] symmetrize(double[] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		
		double[] output;
		double[][] in = new double[1][];
		Matrix result;
		
		in[0]	= input;
		result	= M[0].transpose().times(M[1].times((new Matrix(in)).transpose()));
		output	= result.transpose().getArray()[0];
		return output;
	}
	
	public double[] compressGradient(double[] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		double[] output;
		double[][] in = new double[1][];
		Matrix result;
		
		in[0]	= input;
		result	= M[0].times((new Matrix(in)).transpose());
		output	= result.transpose().getArray()[0];
		return output;
	}
	
	public double[][] compressHessian(double[][] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		Matrix H = M[0].times((new Matrix(input)).times(M[0].transpose()));
		return H.getArrayCopy();
	}
	
	public double[][] uncompressHessian(double[][] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		Matrix H = (M[0].transpose()).times((new Matrix(input)).times(M[0]));
		return H.getArrayCopy();
	}
	
	public double[] compressPositionVector(double[] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		double[] output;
		double[][] in = new double[1][];
		Matrix result;
		
		in[0] = input;
		result = M[1].times((new Matrix(in)).transpose());
		output = result.transpose().getArray()[0];
		return output;
	}
	
	public double[] uncompress(double[] input) {
		if (M==null) {
			return input;
		}
		if (input==null) {
			return null;
		}
		
		double[] output;
		double[][] in = new double[1][];
		Matrix result;
		
		in[0] = input;
		result = M[0].transpose().times((new Matrix(in)).transpose());
		output = result.transpose().getArray()[0];
		return output;
	}
	
	//Central difference method
	//startLoc is FULL dimensionality (NOT symmetry reduced)
	public double[] gradientFiniteDifferences(double[] startLoc, double stepSize) {
		int totFeatures		= this.getTotFeatures();					//Operate in the FULL dimensionality space
		double forwardDifference, reverseDifference;
		double[] baseBetas, modBetas;
		double[] fdGradient	= new double[totFeatures];
		
		baseBetas = Array.clone(this.symmetrize(startLoc));
		//compute function values
		for (int i=0; i<totFeatures; i++) {
			if (this.isSymmetrized(i))		continue;			//Position has already been accounted for (by symmetry)
			try {
				modBetas			= Array.clone(baseBetas);
				modBetas[i]			+= stepSize;
				setParams(symmetrize(modBetas));
				forwardDifference	= functionEval();
				modBetas[i]			-= 2*stepSize;
				setParams(symmetrize(modBetas));
				reverseDifference	= functionEval();
				fdGradient[i]		= (forwardDifference-reverseDifference)/(2*stepSize);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		setParams(startLoc);
		return this.symmetrize(fdGradient);
	}
	
	//Central difference method
	public double[][] hessianFiniteDifferences(double[] startLoc, double stepSize) {
		int totFeatures 			= this.getTotFeatures();
		double[] baseBetas, modBetas;
		double[][] forwardDifference= new double[totFeatures][totFeatures];
		double[][] reverseDifference= new double[totFeatures][totFeatures];
		double[][] fdHessian		= new double[totFeatures][totFeatures];
        CompactGradientOutput output;
		
		baseBetas = Array.clone(startLoc);
		//compute gradients
		for (int i=0; i<totFeatures; i++) {
			try {
				modBetas			= Array.clone(baseBetas);
				modBetas[i]			+= stepSize;
				setParams(modBetas);
				output				= gradientEval();
				//Handle the case where the gradient has not been defined
				if (output==null)	throw new UnsupportedOperationException(
						"The function gradientEval() has not been defined!");
				forwardDifference[i]= output.gradientVector;
				modBetas[i]			-= 2*stepSize;
				setParams(modBetas);
				output				= gradientEval();
				reverseDifference[i]= output.gradientVector;
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		//Find finite differences (forward/backward FD)
		for (int i=0; i<totFeatures; i++) {
			for (int j=0; j<totFeatures; j++) {
				fdHessian[i][j] = (forwardDifference[j][i]-reverseDifference[j][i])/(4*stepSize) + 
									(forwardDifference[i][j]-reverseDifference[i][j])/(4*stepSize);
			}
		}
		setParams(startLoc);
		return fdHessian;
	}
	
	public class CompactGradientOutput {
		public double functionValue		= 0;
		public double[] gradientVector;
		public double[][] hessian;

		public CompactGradientOutput(double functionValue, double[] gradientVector) {
			this.functionValue	= functionValue;
			this.gradientVector	= gradientVector;
		}
		
		public CompactGradientOutput(double[] gradientVector) {
			this.gradientVector	= gradientVector;
		}
		
		public CompactGradientOutput(double functionValue, double[] gradientVector, double[][] hessian) {
			this.functionValue	= functionValue;
			this.gradientVector	= gradientVector;
			this.hessian		= hessian;
		}
		
		public CompactGradientOutput(double[] gradientVector, double[][] hessian) {
			this.gradientVector	= gradientVector;
			this.hessian		= hessian;
		}
		
		public CompactGradientOutput(double[][] hessian) {
			this.hessian		= hessian;
		}
	}
	
public static class bindingModeSymmetry {
		
		public Map<Integer, Boolean> symmetryCMap = new HashMap<>(); //Is the base-feature complement-symmetric?
		public ArrayList<Integer> symmetryIDList = new ArrayList<Integer>(); //IDs of bases along binding site
		public ArrayList<Boolean> symmetryBarrierList = new ArrayList<Boolean>(); //Indicators of barriers across binding site. 
		int k;
		
		/*  FORMAT OF SYMMETRY STRING:
		 *  ==========================
		 *  
		 *  FORMAT 1: Binding block definition
		 *  	- Symmetry string = "Block_1|Block_2;...|Block_N", where for each Block i we have: 
		 *  
		 *              Block_i = "ID_i:L_i:RC_i", where
		 *              
		 *               ID_i : Integer ID of block. Negative integer denotes reverse complement. 
		 *                      ID=0 is gives block with vanishing readout coefficients.
		 *               L_i  : Length of block.
		 *               RC_i : 1=(Reverse-complement symmetric block), 0=(Unconstrainted)
		 *              
		 *      - Example: 1:5:0;0:2:1;-1:5:0  gives a binding model with two identical separated blocks:
		 *           >>>>>NN<<<<<        
		 *                 
		 *  FORMAT 2: Base definition
		 *  	- The symmetry string consists of a series of bases and barriers:
		 *  	- Bases are indicated using letters, numbers, and "."
		 *  	- Each letter defines a set of equivalent bases.
		 *  	- Lower case indicates the complement of upper case
		 *  	- Numbers indicate bases that complement map to themselves (e.g. complement-symmetric).
		 *  	- "." indicate a base with vanishing readout   
		 *  	- Pipe sign "|" indicate barriers that the interactions cannot span.
		 *  	- Example 1: "ABCDE|..|edcba" is the same as above
		 *  	- Example 2: "AB1ba|..|AB1ba" is the same as above but each block is reverse-complement symmetric.
		 *  
		 */		
		public bindingModeSymmetry(String symString, int k) {
			
			if(symString == null || symString.toLowerCase().equals("null")){//Binding mode unconstrained if symString is "null"
				this.k=k;
				for(int ID=1; ID<=k; ID++){
					symmetryCMap.put(ID, false);
					symmetryIDList.add(ID);
					symmetryBarrierList.add(false);
				}
				return;
			}
			
			if(symString.split(":").length>1){
				//USING FORMAT 1 ABOVE
				String[] blockStrings = symString.split("\\|");
				
				Map<Integer, Integer[]> blockVectors = new HashMap<>(); //Is the base-feature complement-symmetric?
				Map<Integer, Boolean> rcSym = new HashMap<>();
				
				int nID=0;//Current number of unique base IDs
				
				for(int i=0; i<blockStrings.length; i++){
					//Parses block string
					String[] entries = blockStrings[i].split(":");
					if(entries.length != 3)
						throw new IllegalArgumentException("Each block must have exactly 3 integer values.");
					int id = Integer.parseInt(entries[0]);
					int L  = Integer.parseInt(entries[1]);
					if(L<=0)
						throw new IllegalArgumentException("Length of block must larger than zero.");
					boolean rc = Integer.parseInt(entries[2]) == 1 ? true : false;
					
					if(rc)id = Math.abs(id); //Can't have reverse of reverse-symmetric block. 
					
					if(id<0 && rcSym.get(-id)!= null && rcSym.get(-id) == true )
						throw new IllegalArgumentException("Blocks are inconsistent.");
					Integer[] currentBlockIDVector;
					if(id == 0) {
						//Uses null-filled block vector if ID=0;
						currentBlockIDVector = new Integer[L];
					} else {
						//Makes sure that the block stored
						if(blockVectors.get(id) == null){
							
							//Adds new block to list of blocks
							if(rc){
								//Adds a reverse-symmetric block.
								Integer[] newBlockVector = new Integer[L];
								blockVectors.put( Math.abs(id), newBlockVector);
								rcSym.put( Math.abs(id), true);
								
								for(int j=0; j<L;j++) {
									//L=4: 0,1<>2,3
									//L=3" 0,<1>2 
									if(j<(L+1)/2){//If we are in the 1st half of the block (including center)
										nID++;
										if(2*j+1 == L) //Checks if the new ID is in the center.
											symmetryCMap.put( nID,  true);
										else {
											symmetryCMap.put( nID, false);
											symmetryCMap.put(-nID, false);
										}
										newBlockVector[j]       =  nID;
									} else {
										newBlockVector[j]       = -(nID-(j-L/2));
									}
								}
							} else {
								//Adds a non-symmetric block.
								Integer[] newBlockVectorRC = new Integer[L];
								Integer[] newBlockVector = new Integer[L];
								blockVectors.put( Math.abs(id), newBlockVector);
								blockVectors.put(-Math.abs(id), newBlockVectorRC);
								rcSym.put( Math.abs(id), false);
								rcSym.put(-Math.abs(id), false);
								
								for(int j=0; j<L;j++) {
									nID++;
									newBlockVector[j]       =  nID;
									newBlockVectorRC[L-1-j] = -nID;
									symmetryCMap.put(nID,  false);
									symmetryCMap.put(-nID, false);
								}
							}
						} else {
							//If the block ID already exists, check so the new and old block are consistent.
							if(rc != rcSym.get(id) || L != blockVectors.get(id).length)
								throw new IllegalArgumentException("Blocks are inconsistent.");
							
						}
						currentBlockIDVector = blockVectors.get(id);
					}
					
					//Adds block to binding mode.
					for(int j=0; j<currentBlockIDVector.length; j++){
						//Adds base IDs to list.
						symmetryIDList.add(currentBlockIDVector[j]);
						//Adds barriers to list (true for last element):
						symmetryBarrierList.add(j==currentBlockIDVector.length-1);
					}
				}
				
			} else {
				//USING INPUT FORMAT 2 
				Map<String, Integer> charIDRep = new HashMap<>();//Numeric representation of character
				
				int nID=0;//Current number of unique IDs 
				for(int i=0; i<symString.length(); i++){
					String current = symString.substring(i,i+1);
					if(current.equals("|")){
						//Records Barrier
						if(nID==0) //Ignores barrier at first position.
							continue;
						if(symmetryBarrierList.size() < symmetryIDList.size())  //Record barrier
							symmetryBarrierList.add(true);
						else
							throw new IllegalArgumentException("Invalid barriers.");
					} else {
						if(symmetryBarrierList.size() < symmetryIDList.size()) //Record lack of barrier.
							symmetryBarrierList.add(false);
						//Parses base data
						if(current.equals(".")){ //Adds empty base.
							symmetryIDList.add(null);
						} else { 
							//Adds new non-trivial base
							if(charIDRep.get(current) == null){
								nID++;
								charIDRep.put(current.toUpperCase(),  nID);
								if(current.toUpperCase().equals(current.toLowerCase()) ) {//Self reverse-complement symmetric
									symmetryCMap.put(charIDRep.get(current.toUpperCase()),true);
								} else {
									charIDRep.put(current.toLowerCase(), -nID);
									symmetryCMap.put(charIDRep.get(current.toUpperCase()),false);
									symmetryCMap.put(charIDRep.get(current.toLowerCase()),false);
									
								}
							}	
							//Records base.
							symmetryIDList.add(charIDRep.get(current));
						}
					}
				}
			}

//			System.out.println(symmetryIDList);
//			System.out.println(symmetryCMap);
//			System.out.println(symmetryBarrierList);
			if (symmetryIDList.size() == k)
				this.k = k;
			else
				throw new IllegalArgumentException("Symmetry string does not match k. L(String)="+symString+", k="+k);
		}
		
		
		//Flattens a list of matrices into a single matrix
		private Matrix flattenMatrix(ArrayList<double[][]> inMatrix) {
			//Computing size of output matrix and location of blocks. 
			if(inMatrix.size()==0 || inMatrix.get(0).length ==0 )
				return null;
			
			int w= inMatrix.get(0)[0].length;
			int h=0;
			ArrayList<Integer> yList = new ArrayList<Integer>(); 
			for(int i=0;i<inMatrix.size();i++){
					if(inMatrix.get(i)[0].length != w)
						throw new IllegalArgumentException("Matrices differ in width.");
					
					yList.add(h);
					h+=inMatrix.get(i).length;
			}
			
			double[][] outMatrix = new double[h][w];
			
			//Copying matrix.
			for(int i=0;i<inMatrix.size();i++)
				for(int y=0; y<inMatrix.get(i).length; y++)
					for(int x=0; x<inMatrix.get(i)[y].length; x++)
						outMatrix[yList.get(i)+y][x] = inMatrix.get(i)[y][x];
			
			return new Matrix(outMatrix);
		}
		
		public ArrayList<Matrix> constructMMatrices() {
			//Creating M-Matrices
			ArrayList<Matrix> mMatrices = new ArrayList<Matrix>();
			
			// Looping over spacings
			for(int dInt=0; dInt<k;dInt++){
//				System.out.println("dInt = "+dInt);
				
				//Creating a list of valid interactions
				//=================================================			
				Map<String, Integer> dinuclRep = new HashMap<>(); // Maps a dinucleotide string (e.g. 1,-2) to an integer 
				Map<Integer, Boolean> rcSymRep  = new HashMap<>();  //Indicates if a dinucletide (integer) is reverse-compllement symmetric. 
				ArrayList<Integer> intIDList   = new ArrayList<Integer>(); //List of interactions in integer notation. 
				
				int nInt = 0; //Number of dinucleotide independent interactions.
				for(int i=0; i<k-dInt; i++){ 
					 //Makes sure both bases non-zero.
					if(symmetryIDList.get(i)==null || symmetryIDList.get(i+dInt)==null){
						intIDList.add(null);
						continue;
					}
					//Makes sure the interaction doesn't span a barrier.
					boolean barrierSpanned = false;
					for(int x=i;x<i+dInt;x++)
						if(symmetryBarrierList.get(x)){
							barrierSpanned = true;
							break;					
						}
					if(barrierSpanned){
						intIDList.add(null);
						continue;
					}
							
					//Creates key strings for the interactions.. 
					int n1=symmetryIDList.get(i);
					int n2=symmetryIDList.get(i+dInt);
					int n1C = symmetryCMap.get(n1) ? n1 : -n1;  //Complement base.
					int n2C = symmetryCMap.get(n2) ? n2 : -n2;
					String intKey   = ""+ n1 +","+ n2;
					String intKeyRC = ""+n2C+"," +n1C;
					
					//Storing information about interaction if it is new. 
					if(dinuclRep.get(intKey) == null){
						nInt++;
						
						//Adding information about the new binding mode 				
						dinuclRep.put(intKey,nInt);
						
						//Determining symmetry of new interaction
						Boolean isRCSym = intKey.equals(intKeyRC);
						rcSymRep.put(nInt, isRCSym);

						//Adding information to RC interaction. 
						if (!isRCSym){
							dinuclRep.put(intKeyRC,-nInt);
							rcSymRep.put(-nInt, isRCSym);
						}			
					}
					
					//Adding information about the current interaction to the binding mode. 
					intIDList.add(dinuclRep.get(intKey));				
				}
				
//				System.out.println("> Valid interactions:");
//				System.out.println(dinuclRep);
//				System.out.println(intIDList);
//				System.out.println(rcSymRep);
				
				//Building matrix:
				//================	
				int nDof = dInt>0 ? 16 : 4;
				ArrayList<double[][]> interactionMatrix = new ArrayList<double[][]>();
				//Creating a list of sub-matrices, one per independent bases.
				for(int id=1; id<=nInt; id++) {
					//Creating submatrices.
					if(dInt==0){
						//2 lines for self-complement bases, 4 for unconstrained.
						interactionMatrix.add(new double[rcSymRep.get(id) ? 2 : 4][nDof*(k-dInt)]);
					} else {
						//10 lines for self-complement dinucleotide, 16 for unconstrained.
						interactionMatrix.add(new double[rcSymRep.get(id) ?10 : 16][nDof*(k-dInt)]);
					}

					
				}

				//Filling in matrix.
				for(int x=0; x<k-dInt; x++) 
					if(intIDList.get(x) != null) {
						if(dInt == 0) { //Mononucleotide
							for(int n=0; n<4; n++) {
								int iRow; //Row index of reduced feature
								if(rcSymRep.get(intIDList.get(x)))  //self-complement base.
									iRow = Math.min(n, nDof-1-n);
								else
									iRow = intIDList.get(x)>0 ? n : nDof-1-n;
								interactionMatrix.get(Math.abs(intIDList.get(x))-1)[iRow][nDof*x+n] = 1;
									
							}
						} else {        //Dinucleotide interactions
							for(int n1=0; n1<4; n1++) {
								for(int n2=0; n2<4; n2++) {
									int iRow; //Row index of reduced feature
									if(rcSymRep.get(intIDList.get(x))) { //Interaction self-complement symmetric.
										iRow = Math.min(4*n1+n2, 4*(3-n2)+(3-n1));
										if(iRow>9)iRow-=2; //Compactivies matrix so all rows are non-zero
										if(iRow>6)iRow-=1;
									} else {
										if(intIDList.get(x)>0) //Forward feature = Diagonal matrix
											iRow=4*n1    +n2;
										else //Reverse feature = RC of identity matrix
											iRow=4*(3-n2)+(3-n1);
									}
//									System.out.println("n1="+n1+", n2="+n2+", iRow = "+iRow+", nDof = "+nDof);
									interactionMatrix.get(Math.abs(intIDList.get(x))-1)[iRow][nDof*x+4*n1+n2] = 1;
								}
							}
							
						}
					}
//				flattenMatrix(interactionMatrix).print(0, 0);
				mMatrices.add(flattenMatrix(interactionMatrix));

			}
			return mMatrices;
		}
	}
}
