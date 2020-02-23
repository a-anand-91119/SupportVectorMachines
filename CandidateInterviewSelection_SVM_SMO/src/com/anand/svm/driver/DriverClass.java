package com.anand.svm.driver;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;

import com.anand.svm.data.DataProvider;
import com.anand.svm.machine.SupportVectorMachine;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;

public class DriverClass extends Application{

	private static final double ZERO = 0.000000009;
	private static SupportVectorMachine supportVectorMachine = null;
	
	public static void main(String[] args) throws IOException {
		String commandLineArgumentsResult = getCommandLineArgumentsResults(args);
		
		double[][] xArray = new double[DataProvider.getTrainingDataSize()][2];
		double[][] yArray = new double[DataProvider.getTrainingDataSize()][1];

		for (int i = 0; i < DataProvider.getTrainingDataSize(); i++) {
			xArray[i][0] = DataProvider.getTrainingData()[i][0][0];
			xArray[i][1] = DataProvider.getTrainingData()[i][0][1];
			yArray[i][0] = DataProvider.getTrainingData()[i][1][0];
		}
		
		switch(commandLineArgumentsResult) {
			case "TRAIN":
				trainModel(xArray, yArray);
				storeModelToFile();
				displayInformationAsTable(xArray, yArray);
				break;
			case "TRAIN_AND_SHOW_RESULT":
				trainModel(xArray, yArray);
				storeModelToFile();
				displayInformationAsTable(xArray, yArray);
				launch();
				break;
			case "CLASSIFY":
				readModelFromFile(args[1]);
				// TODO store alpha in file also
				//displayInformationAsTable(xArray, yArray);
				handleCommandLineEntry();
				break;
		}
		System.exit(0);
	}
	
	private static void readModelFromFile(String modelPath) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(modelPath));
		double[][] weightArray = null;
		String sCurrentLine = null;
		int lineNo = 1;
		double b = 0.0;
		
		while ((sCurrentLine = in.readLine()) != null) {
			switch(lineNo) {
				case 1:
					b = Double.parseDouble(sCurrentLine);
					break;
				case 2:
					weightArray = new double[Integer.parseInt(sCurrentLine)][1];
					break;
				case 3:
					weightArray[0][0] = Double.parseDouble(sCurrentLine);
					break;
				case 4:
					weightArray[1][0] = Double.parseDouble(sCurrentLine);
					break;
			}
			lineNo++;
		}
		in.close();
		supportVectorMachine = new SupportVectorMachine(b, weightArray);
	}

	private static void trainModel(double[][] xArray, double[][] yArray) {
		supportVectorMachine = new SupportVectorMachine(MatrixUtils.createRealMatrix(xArray),
				MatrixUtils.createRealMatrix(yArray));
	}

	private static void storeModelToFile() throws IOException {
		Path path = Paths.get("svm.model");
		
		if(!Files.exists(path, new LinkOption[]{ LinkOption.NOFOLLOW_LINKS}))
			Files.createFile(path);
		
		Files.write(path, String.valueOf(supportVectorMachine.getB() + "\n").getBytes());
		Files.write(path, String.valueOf(supportVectorMachine.getWeight().getData().length  + "\n").getBytes(), StandardOpenOption.APPEND);		
		for(int i = 0;i < supportVectorMachine.getWeight().getData().length;i++)
			Files.write(path, String.valueOf(supportVectorMachine.getWeight().getData()[i][0]  + "\n").getBytes(), StandardOpenOption.APPEND);
	}

	private static String getCommandLineArgumentsResults(String[] args) throws FileNotFoundException {
		if(args.length <= 0) {
			throw new IllegalArgumentException("Mininum Of 1 Argument Is Needed To Use This Program.");
		}
		if(args[0].equals("CLASSIFY")) {
			
			if(args.length < 2)
				throw new IllegalArgumentException("Please Specify The Path Of The Model To Use For Classification");
			
			String modelPath = args[1];
			Path path = Paths.get(modelPath);
			
			if(!Files.exists(path, new LinkOption[]{ LinkOption.NOFOLLOW_LINKS}))
				throw new FileNotFoundException("The Trained Model Not Found In Specified Path: " + modelPath);
		}
		return args[0];
	}

	static void displayInformationAsTable(double[][] fetureArray, double[][] labelsArray) {
		System.out.println("Support Vector 	   | Label   | Alpha");
		printDashes();
		
		for (int i = 0; i < fetureArray.length; i++) {
			if (supportVectorMachine.getAlpha().getData()[i][0] > ZERO
					&& supportVectorMachine.getAlpha().getData()[i][0] != SupportVectorMachine.C) {
				StringBuilder sBuilder = new StringBuilder(String.valueOf(labelsArray[i][0]));
				sBuilder.setLength(5);
				System.out.println(Arrays.toString(fetureArray[i]) + " | " + sBuilder + " | "
						+ new String(String.format("%.10f", supportVectorMachine.getAlpha().getData()[i][0])));
			}
		}
		
		System.out.println("\n		Weight 		| 	b ");
		printDashes();
		System.out.println("<" + new String(String.format("%.9f", supportVectorMachine.getWeight().getData()[0][0]))
				+ ", " + new String(String.format("%.9f", supportVectorMachine.getWeight().getData()[1][0])) + ">	| "
				+ supportVectorMachine.getB());
	}
	
	static void handleCommandLineEntry() throws IOException{
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		System.out.println("Enter (exit) at anytime to stop classification process");
		while(true) {
			System.out.println("\n. Enter scores of Candidates To Classify (Interview 1, Interview 2): ");
			String[] readData = in.readLine().split(" ");
			
			if(readData[0].equalsIgnoreCase("exit"))
				System.exit(0);
			classifyCandidate(readData);	
		}
	}
	
	private static void classifyCandidate(String[] readData) {
		try {
			System.out.println(supportVectorMachine.classify(MatrixUtils.createRealMatrix(
					new double[][] { { Double.valueOf(readData[0]) , Double.valueOf(readData[1]) } })));
		}catch(Exception e) {
			System.out.println("Error Occured: " + e);
			e.printStackTrace();
		}
	}

	static void printDashes() {
		IntStream.range(0, 50).forEach(i -> System.out.print("-"));
		System.out.println();
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		Platform.setImplicitExit(false);

		XYChart.Series<Number, Number> series01 = new XYChart.Series<>();
		XYChart.Series<Number, Number> series02 = new XYChart.Series<>();
		XYChart.Series<Number, Number> series03 = new XYChart.Series<>();
		series01.setName("Candidate Not Hired");
		series02.setName("Candidate Hired");
		series03.setName("Support Vectors");

		IntStream.range(0, DataProvider.getTrainingDataSize()).forEach(i -> {
			double x  = DataProvider.getTrainingData()[i][0][0];
			double y = DataProvider.getTrainingData()[i][0][1];
			
			if (supportVectorMachine.getAlpha().getData()[i][0] > ZERO
					&& supportVectorMachine.getAlpha().getData()[i][0] != SupportVectorMachine.C)
				series03.getData().add(new XYChart.Data<Number, Number>(x, y));
			else if (DataProvider.getTrainingData()[i][1][0] == -1)
				series01.getData().add(new XYChart.Data<Number, Number>(x, y));
			else
				series02.getData().add(new XYChart.Data<Number, Number>(x, y));
		});

		NumberAxis xAxis = new NumberAxis(0, 10, 1.0);
		NumberAxis yAxis = new NumberAxis(0, 10, 1.0);
		xAxis.setLabel("Score For Candidate Interview #1");
		yAxis.setLabel("Score For Candidate Interview #2");
		
		ScatterChart<Number, Number> scatterChart = new ScatterChart<>(xAxis, yAxis);
		scatterChart.getData().add(series01);
		scatterChart.getData().add(series02);
		scatterChart.getData().add(series03);
		
		double m = -(supportVectorMachine.getWeight().getData()[0][0] / supportVectorMachine.getWeight().getData()[1][0]);
		double b = -(supportVectorMachine.getB() / supportVectorMachine.getWeight().getData()[1][0]);
		double score1X = 0.00;
		double score1Y = m * score1X + b;
		double score2X = 10.0;
		double score2Y = m * score2X + b;
		
		XYChart.Series<Number, Number> series04 = new XYChart.Series<>();
		series04.getData().add(new XYChart.Data<Number, Number>(score1X, score1Y));
		series04.getData().add(new XYChart.Data<Number, Number>(score2X, score2Y));
		
		LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);
		lineChart.getData().add(series04);
		lineChart.setOpacity(0.4);
		
		Pane pane = new Pane();
		pane.getChildren().addAll(scatterChart, lineChart);
		primaryStage.setScene(new Scene(pane, 550, 420));
		primaryStage.setOnHidden(e -> {
			try {
				handleCommandLineEntry();
			}catch(Exception ex) {
				ex.printStackTrace();
			}
		});
		System.out.println("\nClose display window to proceed");
		primaryStage.setTitle("Support Vector Machine Test");
		primaryStage.show();
		
	}
}
