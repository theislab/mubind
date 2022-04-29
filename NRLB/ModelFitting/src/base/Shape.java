package base;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import base.Sequence;

public class Shape {
	private static final String[] shapeTypes= {"MGW", "ProT", "HelT", "Roll"};
	private static final int[] maxShapeCols	= {1, 1, 2, 2};
	private String filePath;
	private boolean[] desiredTypes		= null;
	private static int[] shapeDims		= {0, 0, 0, 0};
	public double[][] shapeFeatures		= null;
	public boolean isSet				= false;
	private ArrayList<double[][]> shapes= new ArrayList<double[][]>(4);
	
	public Shape(String folderPath, boolean parseAll) {
		this.filePath = folderPath;
		for (int i=0; i<4; i++) {
			shapes.add(null);
		}
		if (parseAll) {
			try {
				getShape("MGW", "MGW.txt");
				getShape("ProT", "ProT.txt");
				getShape("HelT", "HelT.txt");
				getShape("Roll", "Roll.txt");				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public double[][] setFeatures(String ... types) throws IllegalArgumentException{
		if (types.length>4 || types.length<1) {
			throw new IllegalArgumentException("Invalid Shape Parameters.");
		}
		int nParams 		   = 0;
		boolean[] desiredTypes = new boolean[4];
		double[][] output;
				
		//Identify the desired shape classes
		for (String currType : types) {
			boolean isMatch = false;
			for (int i=0; i<4; i++) {
				if (currType.equals(shapeTypes[i])) {
					desiredTypes[i] = true;
					isMatch = true;
					nParams += shapeDims[i];
					break;
				}
			}
			if (!isMatch) {
				throw new IllegalArgumentException("Invalid Shape Parameter: "+currType);
			}
		}
		if (nParams==0) {
			throw new IllegalArgumentException("Please load shape files.");
		}
		
		//return shape params 
		output = new double[nParams][(int) Math.pow(4, 5)];
		int idx = 0;
		
		for (int currShapeClass=0; currShapeClass<4; currShapeClass++) {
			if (desiredTypes[currShapeClass]) {
				if (shapes.get(currShapeClass)==null) {
					throw new IllegalArgumentException("Please load a "+
							shapeTypes[currShapeClass]+" file.");
				}
				for (int set=0; set<shapeDims[currShapeClass]; set++) {
					output[idx+set] = shapes.get(currShapeClass)[set];
				}
				idx += shapeDims[currShapeClass];
			}
		}
		isSet = true;
		shapeFeatures = Array.transpose(output);
		this.desiredTypes = desiredTypes;		
		return Array.transpose(output);
	}
	
	public double[][] getFeatures() throws IllegalArgumentException {
		if (isSet) {
			return shapeFeatures;
		} else {
			throw new IllegalArgumentException("No Features Set. Please set shape features to use.");
		}
	}
	
	public int nShapeFeatures() throws IllegalArgumentException {
		if (isSet) {
			return shapeFeatures[0].length;
		} else {
			throw new IllegalArgumentException("No Features set. Please set shape features to use.");
		}
	}
	
	public String[] getShapeClasses() throws IllegalArgumentException {
		if (isSet) {
			int nTypes = 0;
			for (boolean i : desiredTypes)	{
				if (i)	nTypes++;
			}
			String[] output = new String[nTypes];
			int currIdx		= 0;
			for (int i=0; i<desiredTypes.length; i++) {
				if (desiredTypes[i]) {
					output[currIdx] = shapeTypes[i];
					currIdx++;
				}
			}
			return output;
		} else {
			throw new IllegalArgumentException("No Features set. Please set shape features to use.");
		}
	}
	
	public double[][] getShape(String type, String fileName) {
		boolean isMatch	= false;
		int shapeIdx	= 0;
		
		//Identify the desired shape
		for (int i=0; i<shapeTypes.length; i++) {
			if (shapeTypes[i].equals(type)) {
				isMatch = true;
				shapeIdx = i;
				break;
			}
		}
		if (!isMatch) {
			throw new IllegalArgumentException("Invalid Shape Parameter: "+type);
		}
		
		//Parse shape file if the shape hasn't been parsed already
		if (shapes.get(shapeIdx)==null) {
			try {
				shapes.set(shapeIdx, fileParser(shapeIdx, fileName));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		shapeDims[shapeIdx] = shapes.get(shapeIdx).length;
		return Array.transpose(shapes.get(shapeIdx));
		
	}
		
	private double[][] fileParser(int shapeIdx, String fileName) 
			throws IOException {
		boolean matchedEntries = false;
		int currSeq, nCols = 0, nEntries = 0;
		String line, seq, shapeName = shapeTypes[shapeIdx];
		String[] parsed;
		double[][] output;		
		BufferedReader br = new BufferedReader(new FileReader(filePath+fileName));
		
		//Read header to see the number of columns
		line	= br.readLine();
		parsed	= line.split(",");
		for (int currEntry=0; currEntry<maxShapeCols[shapeIdx]; currEntry++) {
			if (parsed.length==(currEntry*2+3)) {
				matchedEntries	= true;
				nEntries		= (currEntry*2+3);
				nCols			= currEntry+1;
				break;
			}
		}
		if (!matchedEntries) {
			br.close();
			throw new IOException("Not a "+shapeName+" File/Incorrect Format: "
					+filePath+fileName);			
		}
		output = new double[nCols][(int) Math.pow(4, 5)];
		
		while((line = br.readLine()) != null) {
			parsed = line.split(",");				//split lines in CSV
			if (parsed.length != nEntries) {
				br.close();
				throw new IOException("Not a "+shapeName+" File/Incorrect Format: "
						+filePath+fileName);
			}
			seq = parsed[0].replaceAll("\"", "");	//remove quotes if they exist
			currSeq = (int) (new Sequence(seq, 0, 5)).getValue();
			for (int i=0; i<nCols; i++) {
				output[i][currSeq] = Double.parseDouble(parsed[1+i*2]);
			}
		}
		br.close();
		return output;
	}
}
