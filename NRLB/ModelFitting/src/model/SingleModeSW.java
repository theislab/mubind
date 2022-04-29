package model;

import java.util.Arrays;

import dynamicprogramming.*;
import base.*;

public class SingleModeSW{
	protected boolean isDinuc, isShape;
	protected int l, tLen, flankLength, nShapeClasses, frameOffset, baseOffset;
	public int k, nFeatures, maxFrames;
	protected int nucOffset, dinucOffset, shapeOffset;
	protected long maxFrameValue, frameMask, shapeMask, fFlankingSequence;
	protected long rFlankingSequence, fShapeFlankingSequence, rShapeFlankingSequence;
	protected double[] nucBetas, dinucBetas, shapeBetas;
	protected double[][] shapeFeatures;
	protected DynamicProgramming Z;
	public Round0Model R0Model;

	public SingleModeSW(Shape shapeModel, Round0Model R0, int l, int k, boolean isFlank, 
			int flankLength, String lFlank, String rFlank, boolean isDinuc, boolean isShape) {		
		if (!isFlank || flankLength==0) {
			flankLength = 0;
			isFlank = false;
		}
		
		//Get Round0 Information
		R0Model	 			= R0;
		//Load Shape Table Information
		if (shapeModel!=null) {
			shapeFeatures 	= shapeModel.getFeatures();
			nShapeClasses	= shapeModel.nShapeFeatures();	
		}
		//Load Data
		this.l				= l;
		//Load Parameters
		this.k				= k;
		this.flankLength	= flankLength;
		this.isDinuc		= isDinuc;
		this.isShape		= isShape;

		//Runtime Parameters
		tLen				= (isFlank) ? l+2*flankLength : l;
		maxFrames			= (isFlank) ? l-k+1+2*flankLength : l-k+1;
		maxFrameValue		= (long) Math.pow(4, k);
		frameMask			= maxFrameValue-1;
		shapeMask			= (long) Math.pow(4, k+4)-1;
		baseOffset			= 0;
		nucOffset			= baseOffset+4*k;
		dinucOffset			= nucOffset+16*(k-1);
		shapeOffset			= (isDinuc) ? dinucOffset+nShapeClasses*k : nucOffset+nShapeClasses*k;
		frameOffset			= 0;
		long tempLeft		= (new Sequence(lFlank, 0, lFlank.length())).getValue();
		long tempRight		= (new Sequence(rFlank, 0, rFlank.length())).getValue();
		long tempLeftRC		= reverseComplement(tempRight, rFlank.length());
		long tempRightRC	= reverseComplement(tempLeft, lFlank.length());
		tempRight			= reverse(tempRight, rFlank.length());
		tempRightRC			= reverse(tempRightRC, lFlank.length());
		if (isFlank) {
			long flankMask			= (long) Math.pow(4, flankLength) - 1;
			long shapeFlankMask		= (long) Math.pow(4, flankLength+2) - 1;
			fFlankingSequence		= ((tempLeft & flankMask) << 2*l);
			fFlankingSequence		<<= 2*flankLength;
			fFlankingSequence		= fFlankingSequence | reverse((tempRight & flankMask), flankLength);
			rFlankingSequence		= ((tempLeftRC & flankMask) << 2*l);
			rFlankingSequence		<<= 2*flankLength;
			rFlankingSequence		= rFlankingSequence | reverse((tempRightRC & flankMask), flankLength);
			fShapeFlankingSequence	= ((tempLeft & shapeFlankMask)) << 2*l;
			fShapeFlankingSequence	<<= 2*(flankLength+2);
			fShapeFlankingSequence	= fShapeFlankingSequence | reverse((tempRight & shapeFlankMask), flankLength+2);
			rShapeFlankingSequence	= ((tempLeftRC & shapeFlankMask) << 2*l);
			rShapeFlankingSequence	<<= 2*(flankLength+2);
			rShapeFlankingSequence	= rShapeFlankingSequence | reverse((tempRightRC & shapeFlankMask), flankLength+2);	
		} else {
			fShapeFlankingSequence	= ((tempLeft & 15) << 2*l) << 4;
			fShapeFlankingSequence	= fShapeFlankingSequence | reverse((tempRight & 15), 2);
			rShapeFlankingSequence	= ((tempLeftRC & 15) << 2*l) << 4;
			rShapeFlankingSequence	= rShapeFlankingSequence | reverse((tempRightRC & 15), 2);
		}
		
		nFeatures		+= 4*k;
		if (isDinuc)	nFeatures += 16*(k-1);
		if (isShape)	nFeatures += nShapeClasses*k;
		
		//What type of partition function should be used?
		if (isFlank) {
			if (isDinuc) {
				if (isShape) {
					Z = new FullFeatureNucleotideDinucleotideShape(l, k, false, flankLength, lFlank, rFlank, R0Model, shapeModel);
				} else {
					Z = new FullFeatureNucleotideDinucleotide(l, k, false, flankLength, lFlank, rFlank, R0Model);							
				}
			} else {
				if (isShape) {
					Z = new FullFeatureNucleotideShape(l, k, false, flankLength, lFlank, rFlank, R0Model, shapeModel);							
				} else {
					Z = new FullFeatureNucleotide(l, k, false, flankLength, lFlank, rFlank, R0Model);							
				}
			}			
		} else {
			if (isDinuc) {
				if (isShape) {
					Z = new FullFeatureNucleotideDinucleotideShapeNoFlank(l, k, false, lFlank, rFlank, R0Model, shapeModel);							
				} else {
					Z = new FullFeatureNucleotideDinucleotideNoFlank(l, k, false, lFlank, rFlank, R0Model);							
				}
			} else {
				if (isShape) {
					Z = new FullFeatureNucleotideShapeNoFlank(l, k, false, lFlank, rFlank, R0Model, shapeModel);							
				} else {
					Z = new FullFeatureNucleotideNoFlank(l, k, false, lFlank, rFlank, R0Model);							
				}
			}
		}
		
		setParams(new double[nFeatures]);
	}
	
	public void setOffsets(int baseOffset, int frameOffset) {
		this.baseOffset	= baseOffset;
		nucOffset		= baseOffset+4*k;
		dinucOffset		= nucOffset+16*(k-1);
		shapeOffset		= (isDinuc) ? dinucOffset+nShapeClasses*k : nucOffset+nShapeClasses*k;
		this.frameOffset= frameOffset;
	}
	
	public void setParams(double[] nucBetas, double[] dinucBetas, double[] shapeBetas) {
		double[] dinucAlphas = null;
		
		this.nucBetas	= nucBetas;
		if (isDinuc) {
			this.dinucBetas	= dinucBetas;
			dinucAlphas		= Array.exp(dinucBetas);
		}
		if (isShape)	this.shapeBetas	= shapeBetas;
		Z.setAlphas(Array.exp(nucBetas), dinucAlphas, shapeBetas);
	}
	
	//setParams does NOT SYMMETRIZE input
	public void setParams(double[] input) {
		double[] nucBetas = Arrays.copyOfRange(input, 0, nucOffset-baseOffset);
		double[] dinucBetas, shapeBetas;
		if (isDinuc) {
			dinucBetas = Arrays.copyOfRange(input, nucOffset-baseOffset, dinucOffset-baseOffset);
			shapeBetas = (isShape) ? Arrays.copyOfRange(input, dinucOffset-baseOffset, shapeOffset-baseOffset) : null;
		} else {
			dinucBetas = null;
			shapeBetas = (isShape) ? Arrays.copyOfRange(input, nucOffset-baseOffset, shapeOffset-baseOffset) : null;
		}
		setParams(nucBetas, dinucBetas, shapeBetas);
	}
		
	public double getZ() {
		return Z.recursiveZ();
	}
	
	public double[][] swShapeProfile(long input) {
		int fShapeIdx;
		long fShapeSeq	= fShapeFlankingSequence | (input << 2*(flankLength+2));
		double[][] out	= new double[nShapeClasses][tLen];
		
		for (int j=0; j<tLen; j++) {
			fShapeIdx = (int) (fShapeSeq & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				out[currShapeFeature][tLen-j-1] = shapeFeatures[fShapeIdx][currShapeFeature];
			}
			fShapeSeq >>= 2;
		}
		return out;
	}
	
	public void swNucleotide(long input, double[] kappas) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swGradNucleotide(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				gradients[baseOffset+loc*4+((int) (forwardSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (reverseSubString&3))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swHessianNucleotide(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdIdx, revIdx;
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString, fwdBinSubString, revBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdIdx = baseOffset+loc*4+((int) (forwardSubString&3));
				revIdx = baseOffset+loc*4+((int) (reverseSubString&3));
				for (int subLoc=0; subLoc<k; subLoc++) {
					hessian[fwdIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))] += fSubSum;
					hessian[revIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))] += rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				gradients[fwdIdx] += fSubSum;
				gradients[revIdx] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swNucleotideNoFlank(long input, double[] kappas) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long forwardStrand = input;
		long reverseStrand = reverseComplement(input, l);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swGradNucleotideNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				gradients[baseOffset+loc*4+((int) (forwardSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (reverseSubString&3))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swHessianNucleotideNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdIdx, revIdx;
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString, fwdBinSubString, revBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;

		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdIdx = baseOffset+loc*4+((int) (forwardSubString&3));
				revIdx = baseOffset+loc*4+((int) (reverseSubString&3));
				for (int subLoc=0; subLoc<k; subLoc++) {
					hessian[fwdIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))] += fSubSum;
					hessian[revIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))] += rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				gradients[fwdIdx] += fSubSum;
				gradients[revIdx] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swNucleotideDinucleotide(long input, double[] kappas) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (reverseSubString & 15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swGradNucleotideDinucleotide(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		
		//Calculate gradients		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				gradients[baseOffset+loc*4+((int) (forwardSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (reverseSubString&3))] += rSubSum;
				gradients[nucOffset+loc*16+((int) (forwardSubString&15))] += fSubSum;
				gradients[nucOffset+loc*16+((int) (reverseSubString&15))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			gradients[baseOffset+(k-1)*4+((int) (forwardSubString&3))] += fSubSum;
			gradients[baseOffset+(k-1)*4+((int) (reverseSubString&3))] += rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swHessianNucleotideDinucleotide(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx;
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString, fwdBinSubString, revBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdNucIdx				= baseOffset+loc*4+((int) (forwardSubString&3));
				revNucIdx				= baseOffset+loc*4+((int) (reverseSubString&3));
				fwdDinucIdx				= nucOffset+loc*16+((int) (forwardSubString&15));
				revDinucIdx				= nucOffset+loc*16+((int) (reverseSubString&15));
				gradients[fwdNucIdx]	+= fSubSum;
				gradients[revNucIdx]	+= rSubSum;
				gradients[fwdDinucIdx] 	+= fSubSum;
				gradients[revDinucIdx]	+= rSubSum;
				for (int subLoc=0; subLoc<k-1; subLoc++) {
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;
					hessian[fwdDinucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum;
					hessian[revDinucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum;
					hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				hessian[fwdNucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]			+= fSubSum;
				hessian[revNucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]			+= rSubSum;
				hessian[fwdDinucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]		+= fSubSum;
				hessian[revDinucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]		+= rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fwdBinSubString		= forwardStrand & frameMask;
			revBinSubString		= reverseStrand & frameMask;
			fwdNucIdx			= baseOffset+(k-1)*4+((int) (forwardSubString&3));
			revNucIdx			= baseOffset+(k-1)*4+((int) (reverseSubString&3));
			gradients[fwdNucIdx] += fSubSum;
			gradients[revNucIdx] += rSubSum;
			for (int subLoc=0; subLoc<k-1; subLoc++) {
				hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;
				hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;
				hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum;
				hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum;
				fwdBinSubString >>= 2;
				revBinSubString >>= 2;
			}
			hessian[fwdNucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]	+= fSubSum;
			hessian[revNucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]	+= rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swNucleotideDinucleotideNoFlank(long input, double[] kappas) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long forwardStrand = input;
		long reverseStrand = reverseComplement(input, l);
		
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (forwardSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (forwardSubString & 15))];
				rSubSum += nucBetas[loc*4+((int) (reverseSubString&3))];
				rSubSum	+= dinucBetas[loc*16 + ((int) (reverseSubString&15))];
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (forwardSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (reverseSubString&3))];
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swGradNucleotideDinucleotideNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;

		//Calculate gradients
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				gradients[baseOffset+loc*4+((int) (forwardSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (reverseSubString&3))] += rSubSum;
				gradients[nucOffset+loc*16+((int) (forwardSubString&15))] += fSubSum;
				gradients[nucOffset+loc*16+((int) (reverseSubString&15))] += rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			gradients[baseOffset+(k-1)*4+((int) (forwardSubString&3))] += fSubSum;
			gradients[baseOffset+(k-1)*4+((int) (reverseSubString&3))] += rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swHessianNucleotideDinucleotideNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx;
		double fSubSum, rSubSum;
		long forwardSubString, reverseSubString, fwdBinSubString, revBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		
		//Calculate gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			forwardSubString = forwardStrand & frameMask;
			reverseSubString = reverseStrand & frameMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				fwdBinSubString = forwardStrand & frameMask;
				revBinSubString = reverseStrand & frameMask;
				fwdNucIdx				= baseOffset+loc*4+((int) (forwardSubString&3));
				revNucIdx				= baseOffset+loc*4+((int) (reverseSubString&3));
				fwdDinucIdx				= nucOffset+loc*16+((int) (forwardSubString&15));
				revDinucIdx				= nucOffset+loc*16+((int) (reverseSubString&15));
				gradients[fwdNucIdx]	+= fSubSum;
				gradients[revNucIdx]	+= rSubSum;
				gradients[fwdDinucIdx]	+= fSubSum;
				gradients[revDinucIdx]	+= rSubSum;
				for (int subLoc=0; subLoc<k-1; subLoc++) {
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;
					hessian[fwdDinucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum;
					hessian[revDinucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum;
					hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;
					hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
				}
				hessian[fwdNucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]			+= fSubSum;
				hessian[revNucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]			+= rSubSum;
				hessian[fwdDinucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]		+= fSubSum;
				hessian[revDinucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]		+= rSubSum;
				forwardSubString >>= 2;
				reverseSubString >>= 2;
			}
			fwdBinSubString		= forwardStrand & frameMask;
			revBinSubString		= reverseStrand & frameMask;
			fwdNucIdx			= baseOffset+(k-1)*4+((int) (forwardSubString&3));
			revNucIdx			= baseOffset+(k-1)*4+((int) (reverseSubString&3));
			gradients[fwdNucIdx] += fSubSum;
			gradients[revNucIdx] += rSubSum;
			for (int subLoc=0; subLoc<k-1; subLoc++) {
				hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;
				hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;
				hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum;
				hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum;
				fwdBinSubString >>= 2;
				revBinSubString >>= 2;
			}
			hessian[fwdNucIdx][baseOffset+(k-1)*4+((int) (fwdBinSubString&3))]	+= fSubSum;
			hessian[revNucIdx][baseOffset+(k-1)*4+((int) (revBinSubString&3))]	+= rSubSum;
			forwardStrand >>= 2;
			reverseStrand >>= 2;
		}
	}
	
	public void swNucleotideShape(long input, double[] kappas) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fShapeSeq = fShapeFlankingSequence | (input << 2*(flankLength+2));
		long rShapeSeq = rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swGradNucleotideShape(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				gradients[baseOffset+loc*4+((int) (fSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (rSubString&3))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					gradients[nucOffset+loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					gradients[nucOffset+loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swHessianNucleotideShape(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx, fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, fwdBinSubString, rSubString, revBinSubString, fShapeSubString;
		long fwdShapeBinSubString, rShapeSubString, revShapeBinSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum			= ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {					//Loop over all positions and calculate the gradient
				fwdBinSubString		= forwardStrand & frameMask;
				revBinSubString		= reverseStrand & frameMask;
				fwdShapeBinSubString= fShapeSeq & shapeMask;
				revShapeBinSubString= rShapeSeq & shapeMask;
				fwdNucIdx			= baseOffset+loc*4+((int) (fSubString&3));
				revNucIdx			= baseOffset+loc*4+((int) (rSubString&3));
				fShapeIdx			= (int) (fShapeSubString & 1023);
				rShapeIdx			= (int) (rShapeSubString & 1023);
				shapePosIdx			= loc*nShapeClasses+nucOffset;
				gradients[fwdNucIdx] += fSubSum;
				gradients[revNucIdx] += rSubSum;
				for (int csc=0; csc<nShapeClasses; csc++) {
					gradients[shapePosIdx+csc]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					gradients[shapePosIdx+csc]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+nucOffset;
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;		//nuc on nuc
					for (int csc=0; csc<nShapeClasses; csc++) {												//shape on nuc
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swNucleotideShapeNoFlank(long input, double[] kappas) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long forwardStrand 	= input;
		long reverseStrand 	= reverseComplement(input, l);
		long fShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long rShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swGradNucleotideShapeNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				gradients[baseOffset+loc*4+((int) (fSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (rSubString&3))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					gradients[nucOffset+loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					gradients[nucOffset+loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swHessianNucleotideShapeNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx, fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, fwdBinSubString, rSubString, revBinSubString, fShapeSubString;
		long fwdShapeBinSubString, rShapeSubString, revShapeBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {					//Loop over all positions and calculate the gradient
				fwdBinSubString		= forwardStrand & frameMask;
				revBinSubString		= reverseStrand & frameMask;
				fwdShapeBinSubString= fShapeSeq & shapeMask;
				revShapeBinSubString= rShapeSeq & shapeMask;
				fwdNucIdx			= baseOffset+loc*4+((int) (fSubString&3));
				revNucIdx			= baseOffset+loc*4+((int) (rSubString&3));
				fShapeIdx			= (int) (fShapeSubString & 1023);
				rShapeIdx			= (int) (rShapeSubString & 1023);
				shapePosIdx			= loc*nShapeClasses+nucOffset;
				gradients[fwdNucIdx] += fSubSum;
				gradients[revNucIdx] += rSubSum;
				for (int csc=0; csc<nShapeClasses; csc++) {
					gradients[shapePosIdx+csc]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					gradients[shapePosIdx+csc]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+nucOffset;
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;		//nuc on nuc
					for (int csc=0; csc<nShapeClasses; csc++) {														//shape on nuc
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swNucleotideDinucleotideShape(long input, double[] kappas) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long forwardStrand = fFlankingSequence | (input << 2*flankLength);
		long reverseStrand = rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fShapeSeq = fShapeFlankingSequence | (input << 2*(flankLength+2));
		long rShapeSeq = rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swGradNucleotideDinucleotideShape(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				gradients[baseOffset+loc*4+((int) (fSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (rSubString&3))] += rSubSum;
				gradients[nucOffset+loc*16+((int) (fSubString&15))] += fSubSum;
				gradients[nucOffset+loc*16+((int) (rSubString&15))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					gradients[dinucOffset+loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					gradients[dinucOffset+loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			gradients[baseOffset+(k-1)*4+((int) (fSubString&3))] += fSubSum;
			gradients[baseOffset+(k-1)*4+((int) (rSubString&3))] += rSubSum;
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				gradients[dinucOffset+(k-1)*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
				gradients[dinucOffset+(k-1)*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swHessianNucleotideDinucleotideShape(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx, fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, fwdBinSubString, rSubString, revBinSubString, fShapeSubString;
		long fwdShapeBinSubString, rShapeSubString, revShapeBinSubString;		
		long fwdStrand			= fFlankingSequence | (input << 2*flankLength);
		long revStrand			= rFlankingSequence | (reverseComplement(input, l) << 2*flankLength);
		long fwdShapeSeq		= fShapeFlankingSequence | (input << 2*(flankLength+2));
		long revShapeSeq		= rShapeFlankingSequence | (reverseComplement(input, l) << 2*(flankLength+2));
		long forwardStrand		= fwdStrand;
		long reverseStrand 		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum			= ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString			= forwardStrand & frameMask;
				revBinSubString			= reverseStrand & frameMask;
				fwdShapeBinSubString	= fShapeSeq & shapeMask;
				revShapeBinSubString	= rShapeSeq & shapeMask;
				fwdNucIdx				= baseOffset+loc*4+((int) (fSubString&3));
				revNucIdx				= baseOffset+loc*4+((int) (rSubString&3));
				fwdDinucIdx				= nucOffset+loc*16+((int) (fSubString&15));
				revDinucIdx				= nucOffset+loc*16+((int) (rSubString&15));
				fShapeIdx				= (int) (fShapeSubString & 1023);
				rShapeIdx				= (int) (rShapeSubString & 1023);				
				shapePosIdx				= loc*nShapeClasses+dinucOffset;
				gradients[fwdNucIdx]	+= fSubSum;
				gradients[revNucIdx]	+= rSubSum;
				if (loc < k-1) {
					gradients[fwdDinucIdx]	+= fSubSum;
					gradients[revDinucIdx]	+= rSubSum;
				}
				for (int csc=0; csc<nShapeClasses; csc++) {
					gradients[shapePosIdx+csc]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					gradients[shapePosIdx+csc]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+dinucOffset;
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]			+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]			+= rSubSum;		//nuc on nuc
					if (loc < k-1) {
						hessian[fwdDinucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum;		//dinuc on nuc
						hessian[revDinucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum;		//dinuc on nuc
					}
					for (int csc=0; csc<nShapeClasses; csc++) {											//shape on nuc
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					if (subLoc < k-1) {
						hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//nuc on dinuc
						hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//nuc on dinuc
						if (loc < k-1) {
							hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//dinuc on dinuc
							hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//dinuc on dinuc
						}
						for (int csc=0; csc<nShapeClasses; csc++) {		//shape on dinuc
							hessian[shapePosIdx+csc][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum*shapeFeatures[rShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					if (loc < k-1) {
						for (int csc=0; csc<nShapeClasses; csc++) {		//dinuc on shape
							hessian[fwdDinucIdx][shapeSubPosIdx+csc]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
							hessian[revDinucIdx][shapeSubPosIdx+csc]	+= rSubSum*shapeFeatures[revShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {		//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}				
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
		
	public void swNucleotideDinucleotideShapeNoFlank(long input, double[] kappas) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long forwardStrand 	= input;
		long reverseStrand 	= reverseComplement(input, l);
		long fShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long rShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = 0;
			rSubSum = 0;
			for (int loc=0; loc<k-1; loc++) {
				fSubSum += nucBetas[loc*4+((int) (fSubString&3))];
				rSubSum += nucBetas[loc*4+((int) (rSubString&3))];
				fSubSum += dinucBetas[loc*16 + ((int) (fSubString & 15))];
				rSubSum += dinucBetas[loc*16 + ((int) (rSubString & 15))];
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					fSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
					rSubSum += shapeBetas[loc*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			} 
			fSubSum += nucBetas[(k-1)*4+((int) (fSubString&3))];
			rSubSum += nucBetas[(k-1)*4+((int) (rSubString&3))];
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				fSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[fShapeIdx][currShapeFeature];
				rSubSum += shapeBetas[(k-1)*nShapeClasses + currShapeFeature]*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			kappas[frameOffset+j*2]		= Math.exp(fSubSum);
			kappas[frameOffset+j*2+1]	= Math.exp(rSubSum);
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swGradNucleotideDinucleotideShapeNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients) {
		int fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, rSubString, fShapeSubString, rShapeSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;
		
		//Calculate Gradients
		for (int j=0; j<maxFrames; j++) {
			fSubString = forwardStrand & frameMask;
			rSubString = reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum = ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum = ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k-1; loc++) {
				gradients[baseOffset+loc*4+((int) (fSubString&3))] += fSubSum;
				gradients[baseOffset+loc*4+((int) (rSubString&3))] += rSubSum;
				gradients[nucOffset+loc*16+((int) (fSubString&15))] += fSubSum;
				gradients[nucOffset+loc*16+((int) (rSubString&15))] += rSubSum;
				fShapeIdx = (int) (fShapeSubString & 1023);
				rShapeIdx = (int) (rShapeSubString & 1023);
				for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
					gradients[dinucOffset+loc*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
					gradients[dinucOffset+loc*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
				}
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}
			gradients[baseOffset+(k-1)*4+((int) (fSubString&3))] += fSubSum;
			gradients[baseOffset+(k-1)*4+((int) (rSubString&3))] += rSubSum;
			fShapeIdx = (int) (fShapeSubString & 1023);
			rShapeIdx = (int) (rShapeSubString & 1023);
			for (int currShapeFeature=0; currShapeFeature<nShapeClasses; currShapeFeature++) {
				gradients[dinucOffset+(k-1)*nShapeClasses + currShapeFeature] += fSubSum*shapeFeatures[fShapeIdx][currShapeFeature];
				gradients[dinucOffset+(k-1)*nShapeClasses + currShapeFeature] += rSubSum*shapeFeatures[rShapeIdx][currShapeFeature];
			}
			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}
	
	public void swHessianNucleotideDinucleotideShapeNoFlank(long input, int ki, double totalSum, double[] kappas, double[] gradients, double[][] hessian) {
		int fwdNucIdx, revNucIdx, fwdDinucIdx, revDinucIdx, fwdShapeIdx, revShapeIdx, shapePosIdx, shapeSubPosIdx, fShapeIdx, rShapeIdx;
		double fSubSum, rSubSum;
		long fSubString, fwdBinSubString, rSubString, revBinSubString, fShapeSubString;
		long fwdShapeBinSubString, rShapeSubString, revShapeBinSubString;
		long fwdStrand			= input;
		long revStrand			= reverseComplement(input, l);
		long fwdShapeSeq 		= fShapeFlankingSequence | (input << 4);
		long revShapeSeq 		= rShapeFlankingSequence | (reverseComplement(input, l) << 4);
		long forwardStrand		= fwdStrand;
		long reverseStrand		= revStrand;
		long fShapeSeq			= fwdShapeSeq;
		long rShapeSeq			= revShapeSeq;

		//Calculate Gradients and hessian term 1
		for (int j=0; j<maxFrames; j++) {
			fSubString		= forwardStrand & frameMask;
			rSubString		= reverseStrand & frameMask;
			fShapeSubString = fShapeSeq & shapeMask;
			rShapeSubString = rShapeSeq & shapeMask;
			fSubSum			= ki*kappas[frameOffset+2*j]/totalSum;
			rSubSum			= ki*kappas[frameOffset+2*j+1]/totalSum;
			for (int loc=0; loc<k; loc++) {
				fwdBinSubString			= forwardStrand & frameMask;
				revBinSubString			= reverseStrand & frameMask;
				fwdShapeBinSubString	= fShapeSeq & shapeMask;
				revShapeBinSubString	= rShapeSeq & shapeMask;
				fwdNucIdx				= baseOffset+loc*4+((int) (fSubString&3));
				revNucIdx				= baseOffset+loc*4+((int) (rSubString&3));
				fwdDinucIdx				= nucOffset+loc*16+((int) (fSubString&15));
				revDinucIdx				= nucOffset+loc*16+((int) (rSubString&15));
				fShapeIdx				= (int) (fShapeSubString & 1023);
				rShapeIdx				= (int) (rShapeSubString & 1023);				
				shapePosIdx				= loc*nShapeClasses+dinucOffset;
				gradients[fwdNucIdx]	+= fSubSum;
				gradients[revNucIdx]	+= rSubSum;
				if (loc < k-1) {
					gradients[fwdDinucIdx]	+= fSubSum;
					gradients[revDinucIdx]	+= rSubSum;
				}
				for (int csc=0; csc<nShapeClasses; csc++) {
					gradients[shapePosIdx+csc]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
					gradients[shapePosIdx+csc]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
				}
				for (int subLoc=0; subLoc<k; subLoc++) {	//Now, for each position look at the contributions from all other positions (cross terms) for hessian
					fwdShapeIdx		= (int) (fwdShapeBinSubString & 1023);
					revShapeIdx		= (int) (revShapeBinSubString & 1023);
					shapeSubPosIdx	= subLoc*nShapeClasses+dinucOffset;
					hessian[fwdNucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]				+= fSubSum;		//nuc on nuc
					hessian[revNucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]				+= rSubSum;		//nuc on nuc
					if (loc < k-1) {
						hessian[fwdDinucIdx][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]		+= fSubSum;		//dinuc on nuc
						hessian[revDinucIdx][baseOffset+subLoc*4+((int) (revBinSubString&3))]		+= rSubSum;		//dinuc on nuc
					}
					for (int csc=0; csc<nShapeClasses; csc++) {														//shape on nuc
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (fwdBinSubString&3))]	+= fSubSum*shapeFeatures[fShapeIdx][csc];
						hessian[shapePosIdx+csc][baseOffset+subLoc*4+((int) (revBinSubString&3))]	+= rSubSum*shapeFeatures[rShapeIdx][csc];
					}
					if (subLoc < k-1) {
						hessian[fwdNucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum;		//nuc on dinuc
						hessian[revNucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum;		//nuc on dinuc
						if (loc < k-1) {
							hessian[fwdDinucIdx][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]	+= fSubSum;		//dinuc on dinuc
							hessian[revDinucIdx][subLoc*16+((int) (revBinSubString&15))+nucOffset]	+= rSubSum;		//dinuc on dinuc
						}
						for (int csc=0; csc<nShapeClasses; csc++) {		//shape on dinuc
							hessian[shapePosIdx+csc][subLoc*16+((int) (fwdBinSubString&15))+nucOffset]		+= fSubSum*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][subLoc*16+((int) (revBinSubString&15))+nucOffset]		+= rSubSum*shapeFeatures[rShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {				//nuc on shape
						hessian[fwdNucIdx][shapeSubPosIdx+csc]				+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
						hessian[revNucIdx][shapeSubPosIdx+csc]				+= rSubSum*shapeFeatures[revShapeIdx][csc];
					}
					if (loc < k-1) {
						for (int csc=0; csc<nShapeClasses; csc++) {			//dinuc on shape
							hessian[fwdDinucIdx][shapeSubPosIdx+csc]		+= fSubSum*shapeFeatures[fwdShapeIdx][csc];
							hessian[revDinucIdx][shapeSubPosIdx+csc]		+= rSubSum*shapeFeatures[revShapeIdx][csc];
						}
					}
					for (int csc=0; csc<nShapeClasses; csc++) {				//shape on shape
						for (int csc2=0; csc2<nShapeClasses; csc2++) {
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= fSubSum*shapeFeatures[fwdShapeIdx][csc2]*shapeFeatures[fShapeIdx][csc];
							hessian[shapePosIdx+csc][shapeSubPosIdx+csc2]	+= rSubSum*shapeFeatures[revShapeIdx][csc2]*shapeFeatures[rShapeIdx][csc];
						}
					}
					fwdBinSubString >>= 2;
					revBinSubString >>= 2;
					fwdShapeBinSubString >>= 2;
					revShapeBinSubString >>= 2;
				}				
				fSubString >>= 2;
				rSubString >>= 2;
				fShapeSubString >>= 2;
				rShapeSubString >>= 2;
			}			
			forwardStrand >>= 2;
			reverseStrand >>= 2;
			fShapeSeq >>= 2;
			rShapeSeq >>= 2;
		}
	}

	public void reverseBetas() {
		nucBetas 		= Array.blockReverse(nucBetas, 4);
		if (isDinuc) 	dinucBetas 	= Array.blockReverse(dinucBetas, 16);	
		if (isShape)	shapeBetas 	= Array.blockReverse(shapeBetas, nShapeClasses);
	}
	
	private long reverse(long input, int length) {
		long output = 0;
		
		for (int i=0; i<length; i++) {
			output = ((output << 2) | (input & 3));
			input = input >> 2;
		}
		return output;
	}
	
	private long reverseComplement(long input, int length) {
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