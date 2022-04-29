package base;

import java.io.Serializable;

public abstract class Results implements Serializable {
	private static final long serialVersionUID = -4967438232867150674L;

	//Add a fit
	public abstract void addFit(Fit in); 
	
	//Read a serialized file
	public abstract void read(String fileName) throws Exception;
	
	//Provide an equals method
	public abstract boolean equals(Object compareTo); 
	
	//Store object in a file
	public abstract void store(String fileName, boolean isAppend) throws Exception;
}
