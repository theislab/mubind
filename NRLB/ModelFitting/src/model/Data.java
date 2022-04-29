package model;

public class Data {
	public int l;
	public int nCount;
	public String leftFlank;
	public String rightFlank;
	public int[] counts;
	public long[] probes;
	public double[] R0Prob;
	public Round0Model R0Model;

	public Data(int l, int nCount, String leftFlank, String rightFlank, int[] counts, 
			long[] probes, double[] R0Prob, Round0Model R0Model) {
		this.l			= l;
		this.nCount		= nCount;
		this.leftFlank	= leftFlank;
		this.rightFlank = rightFlank;
		this.counts		= counts;
		this.probes		= probes;
		this.R0Prob		= R0Prob;
		this.R0Model	= R0Model;
	}
	
	@Override
	public Data clone() {
		return (new Data(l, nCount, leftFlank, rightFlank, counts, probes,
				R0Prob, R0Model));
	}
}
