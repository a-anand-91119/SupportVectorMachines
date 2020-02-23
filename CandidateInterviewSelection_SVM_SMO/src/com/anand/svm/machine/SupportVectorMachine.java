package com.anand.svm.machine;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SupportVectorMachine {

	private RealMatrix x;
	private RealMatrix y;
	private RealMatrix alpha;
	private RealMatrix weight;
	private static final int MAX_NUMBER_OF_ITERATIONS = 50;
	private double b = 0;
	private static final double EPSILON = 0.001;
	private static final double MIN_ALPHA_OPTIMIZATION = 0.00001;
	public static final double C = 5.0;

	public SupportVectorMachine(double b, double[][] weightArray) {
		this.b = b;
		this.weight = MatrixUtils.createRealMatrix(weightArray);
	}
	
	public SupportVectorMachine(RealMatrix x, RealMatrix y) {
		this.x = x;
		this.y = y;

		System.out.println("Creating SVM");
		prepareAlphaMatrix();
		performOptimization();
		
		weight = calcuateWeight();
	}

	private void performOptimization() {
		int i = 0;
		System.out.println("Started SVM Training");
		while (i < MAX_NUMBER_OF_ITERATIONS) {
			int noOfAlphaPairsOptimized = performSequentialMinimalOptimization();
			if (noOfAlphaPairsOptimized == 0)
				i++;
			else {
				//System.out.println("Number Of Alpha Pairs Optimized: " + noOfAlphaPairsOptimized);
				/*System.out.println("---------------------------------");
				for(int ix = 0; ix<alpha.getData().length;ix++)
				System.out.println(Arrays.toString(alpha.getData()[ix]));
				System.out.println("*********");
				System.out.println("B: " + b);
				System.out.println("*********");
				System.out.println("---------------------------------");*/
				i = 0;
			}
		}
		System.out.println("SVM Model Created");
	}

	private void prepareAlphaMatrix() {
		double[] alphaArray = new double[x.getData().length];
		IntStream.range(0, alphaArray.length).forEach(i -> alphaArray[i] = 0);
		alpha = MatrixUtils.createColumnRealMatrix(alphaArray);
	}

	private int performSequentialMinimalOptimization() {
		int numberOfAlphaPairsOptimized = 0;
		for (int i = 0; i < x.getData().length; i++) {
			RealMatrix Ei = mult(y, alpha).transpose()
										.multiply(x.multiply(x.getRowMatrix(i).transpose()))
										.scalarAdd(b)
										.subtract(y.getRowMatrix(i));
			
			if(checkIfAlphaViolatesKarushKuhnTuckerConditions(alpha.getEntry(i, 0), Ei.getEntry(0, 0))) {
				int j = selectIndexOfSecondsAlphaToOptimize(i, x.getData().length);
				
				RealMatrix Ej = mult(y, alpha).transpose()
											.multiply(x.multiply(x.getRowMatrix(j).transpose()))
											.scalarAdd(b)
											.subtract(y.getRowMatrix(j));
				
				double oldAplhaI = alpha.getRowMatrix(i).getEntry(0, 0);
				double oldAlphaJ = alpha.getRowMatrix(j).getEntry(0, 0);
				
				double[] bounds = boundAlpha(alpha.getEntry(i, 0), 
											alpha.getEntry(j, 0), 
											y.getEntry(i, 0),
											y.getEntry(j, 0));
				
				double ETA = x.getRowMatrix(i).multiply(x.getRowMatrix(j).transpose()).scalarMultiply(2.0).getEntry(0, 0)
							- x.getRowMatrix(i).multiply(x.getRowMatrix(j).transpose()).getEntry(0, 0)
							- x.getRowMatrix(j).multiply(x.getRowMatrix(j).transpose()).getEntry(0, 0);
				
				if(bounds[0] != bounds[1] && ETA < 0) {
					if(optimizeAlphaPair(i, j, Ei.getEntry(0, 0), Ej.getEntry(0, 0), ETA, bounds, oldAplhaI, oldAlphaJ)) {
						optimizeB(i, j, Ei.getEntry(0, 0), Ej.getEntry(0, 0), oldAplhaI, oldAlphaJ);
						numberOfAlphaPairsOptimized += 1;
					}
				}
			}
		}
		return numberOfAlphaPairsOptimized;
	}
	
	public String classify(RealMatrix entry) {
		String classification = "Classified as -1. (Will not be hired as per prediction)";
		if(Math.signum(entry.multiply(weight).getEntry(0, 0) + b) == 1)
			classification = "Classied as 1. (Will be hired as per prediction)";
		return classification;
	}
	
	private RealMatrix calcuateWeight() {
		double[][] weightArray = new double[x.getData()[0].length][1];
		IntStream.range(0, weightArray.length).forEach(i -> weightArray[i][0] = 0.0);
		RealMatrix weightMatrix = MatrixUtils.createRealMatrix(weightArray);
		
		for(int i =0;i<x.getData().length;i++) {
			weightMatrix = weightMatrix.add(x.getRowMatrix(i).transpose()
					.scalarMultiply(y.getRowMatrix(i).multiply(alpha.getRowMatrix(i)).getEntry(0, 0)));
		}
		
		return weightMatrix;
	}
	
	private boolean optimizeAlphaPair(int i, int j, double Ei, double Ej, double ETA, double[] bounds, double oldAplhaI,
			double oldAlphaJ) {
		
		boolean flag = false;
		alpha.setEntry(j, 0, alpha.getEntry(j, 0) - y.getEntry(j, 0) * (Ei - Ej) / ETA);
		
		clipAlphaJ(j, bounds[1], bounds[0]);
		
		if(Math.abs(alpha.getEntry(j, 0) - oldAlphaJ) >= MIN_ALPHA_OPTIMIZATION) {
			optimizeAlphaISameAsAlphaJInOppositeDirection(i, j, oldAlphaJ);
			flag = true;
		}
		
		return flag;
	}
	
	private void optimizeB(int i, int j, double Ei, double Ej, double oldAplhaI, double oldAlphaJ) {
		double b1 = b - Ei 
				- mult(y.getRowMatrix(i), alpha.getRowMatrix(i).scalarAdd(-oldAplhaI))
				.multiply(x.getRowMatrix(i).multiply(x.getRowMatrix(i).transpose())).getEntry(0, 0)
				- mult(y.getRowMatrix(j), alpha.getRowMatrix(j).scalarAdd(-oldAlphaJ))
				.multiply(x.getRowMatrix(i).multiply(x.getRowMatrix(j).transpose())).getEntry(0, 0);
		
		double b2 = b - Ej 
				- mult(y.getRowMatrix(i), alpha.getRowMatrix(i).scalarAdd(-oldAplhaI))
				.multiply(x.getRowMatrix(i).multiply(x.getRowMatrix(j).transpose())).getEntry(0, 0)
				- mult(y.getRowMatrix(j), alpha.getRowMatrix(j).scalarAdd(-oldAlphaJ))
				.multiply(x.getRowMatrix(j).multiply(x.getRowMatrix(j).transpose())).getEntry(0, 0);
		
		if(0 < alpha.getRowMatrix(i).getEntry(0, 0) && C > alpha.getRowMatrix(i).getEntry(0, 0))
			b = b1;
		else if(0 < alpha.getRowMatrix(j).getEntry(0, 0) && C > alpha.getRowMatrix(j).getEntry(0, 0))
			b = b2;
		else
			b = (b1 + b2) / 2.0;
	}
	
	private void optimizeAlphaISameAsAlphaJInOppositeDirection(int i, int j, double oldAlphaJ) {
		alpha.setEntry(i, 0, 
				alpha.getEntry(i, 0) + y.getEntry(j, 0) * y.getEntry(i, 0) * (oldAlphaJ - alpha.getEntry(j, 0)));
	}
	
	private void clipAlphaJ(int index, double highBound, double lowBound) {
		if(alpha.getEntry(index, 0) < lowBound)
			alpha.setEntry(index, 0, lowBound);
		
		if(alpha.getEntry(index, 0) > highBound)
			alpha.setEntry(index, 0, highBound);
	}
	
	private double[] boundAlpha(double alphaI, double alphaJ, double yI, double yJ) {
		double[] bounds = new double[2];
		if(yI == yJ) {
			bounds[0] = Math.max(0, alphaJ + alphaI - C);
			bounds[1] = Math.min(C, alphaJ + alphaI);
		}else {
			bounds[0] = Math.max(0, alphaJ - alphaI);
			bounds[1] = Math.min(C, alphaJ - alphaI + C);
		}
		return bounds;
	}
	
	private boolean checkIfAlphaViolatesKarushKuhnTuckerConditions(double alpha, double e) {
		return (alpha > 0 && Math.abs(e) < EPSILON) || (alpha < C && Math.abs(e) > EPSILON); 
	}
	
	private int selectIndexOfSecondsAlphaToOptimize(int indexOfFirstAlpha, int numberOfRows) {
		int indexOfSecondAlpha = indexOfFirstAlpha;
		while(indexOfSecondAlpha == indexOfFirstAlpha)
			indexOfSecondAlpha = ThreadLocalRandom.current().nextInt(0, numberOfRows - 1);			
		return indexOfSecondAlpha;
	}

	private static RealMatrix mult(RealMatrix matrix1, RealMatrix matrix2) {
		double[][] returnData = new double[matrix1.getData().length][matrix1.getData()[0].length];
		IntStream.range(0, matrix1.getData().length).forEach(r -> 
			IntStream.range(0, matrix1.getData()[0].length).forEach(c -> 
				returnData[r][c] = matrix1.getEntry(r, c) * matrix2.getEntry(r, c)));
		return MatrixUtils.createRealMatrix(returnData);
	}

	public RealMatrix getAlpha() {
		return alpha;
	}

	public RealMatrix getWeight() {
		return weight;
	}

	public double getB() {
		return b;
	}
	
}
