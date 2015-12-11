import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * k-NN program that performs Cross Validation
 * @author Sujith Vidanapathirana
 * CS 4780 Projectx
 */
public class Knn
{
	/**
	 * CHANGE ME - data files
	 */
	private static final String BINARY_DATA_FILE_V5 = "data/binaryVectorData_v5.txt";
	private static final String TF_IDF_DATA_FILE_V5 = "data/tfIdfVectorData_v5.txt";

	private static final String BINARY_DATA_FILE_V6 = "data/binaryVectorData_v6.txt";
	private static final String TF_IDF_DATA_FILE_V6 = "data/tfIdfVectorData_v6.txt";
	
	private static final String BINARY_DATA_FILE_V7 = "data/binaryVectorData_v7.txt";
	private static final String TF_IDF_DATA_FILE_V7 = "data/tfIdfVectorData_v7.txt";
	
	private static int NUM_EXAMPLES;
	private static int NUM_ATTRIBUTES; //Includes only features  & no labels 
	private static int[] k_values; //k values considering
	private static final int EMOTION_ABSENT =  0;
	private static final int EMOTION_POSITIVE = 1;
	private static final int EMOTION_NEGATIVE = 2;
	private static final int EMOTION_UNCERTAIN = 3;

	/**
	 * For Precision - Recall
	 */
	private static int tp_Absent =0;
	private static  int fp_Absent =0;
	private static  int tp_Positive =0;
	private static  int fp_Positive =0;
	private static  int tp_Negative =0;
	private static  int fp_Negative =0;
	private static  int tp_Uncertain =0;
	private static  int fp_Uncertain =0;
		
	//Structure storing a Feature
	static class Feature
	{
		public int index;  //The ID of the attribute
		public double value;
		
		Feature(int i, double v)
		{	
			this.index = i;
			this.value = v;
		}
	}
	private static Feature[][] trainDataAr;
	
	public static void main(String args[])
	{
		try
		{
			/**
			 * CHANGE ME - K values to consider
			 */
			k_values = new int[200];
			
			for(int i=0; i < 200 ; i++)
				k_values[i] = i+1;
			
			long time = System.currentTimeMillis();

			calculateDimensions(BINARY_DATA_FILE_V5);
			System.out.println("Num Examples: " + NUM_EXAMPLES + "\nNum Attributes: " + NUM_ATTRIBUTES);
			System.out.println("Binary Data Stats V5....");
			performKnn(BINARY_DATA_FILE_V5);
			System.out.println("Tf-idf Data Stats V5....");
			performKnn(TF_IDF_DATA_FILE_V5);
			
			calculateDimensions(BINARY_DATA_FILE_V6);
			System.out.println("Num Examples: " + NUM_EXAMPLES + "\nNum Attributes: " + NUM_ATTRIBUTES);
			System.out.println("Binary Data Stats V6....");
			performKnn(BINARY_DATA_FILE_V6);
			System.out.println("Tf-idf Data Stats V6....");
			performKnn(TF_IDF_DATA_FILE_V6);
			
			calculateDimensions(BINARY_DATA_FILE_V7);
			System.out.println("Num Examples: " + NUM_EXAMPLES + "\nNum Attributes: " + NUM_ATTRIBUTES);
			System.out.println("Binary Data Stats V7....");
			performKnn(BINARY_DATA_FILE_V7);
			System.out.println("Tf-idf Data Stats V7....");
			performKnn(TF_IDF_DATA_FILE_V7);
			
			time = System.currentTimeMillis() - time;
			System.out.println("Total time taken: " + time);
		}
		catch(Exception e)
		{
			System.err.println(e.getMessage());
		}
	}
	
	public static void performKnn(String fileName)
	{
		trainDataAr = new Feature[NUM_EXAMPLES][];
		
		readDataFromFile(fileName);
		
		double[] meanAccuracyAr = performCrossValidation();
		
		List<KvalAccuracyTuple> sortedMeanResultsList = sortArrayByAccuracy(meanAccuracyAr);

		for(KvalAccuracyTuple tuple : sortedMeanResultsList)
		{
			System.out.println(tuple.getK() + "," + tuple.getAccuracy());
		}
		System.out.println("tp_Absent: " + tp_Absent);
		System.out.println("tp_Positive: " + tp_Positive);
		System.out.println("tp_Negative: " + tp_Negative);
		System.out.println("tp_Uncertain: " + tp_Uncertain);

		System.out.println("fp_Absent: " + fp_Absent);
		System.out.println("fp_Positive: " + fp_Positive);
		System.out.println("fp_Negative: " + fp_Negative);
		System.out.println("fp_Uncertain: " + fp_Uncertain);
		
		System.out.println("END of Method===================");
	}
	
	/**
	 * Input: Array of doubles with elements representing accuracy and indices 
	 * representing K values.
	 * Sorts the array by accuracy.
	 */
	public static List<KvalAccuracyTuple> sortArrayByAccuracy(double[] resultsMeanAr)
	{
		List<KvalAccuracyTuple> meanResultList = new ArrayList<KvalAccuracyTuple>();
		
		for(int i =0; i<resultsMeanAr.length; i++)
		{
			KvalAccuracyTuple vo = new KvalAccuracyTuple(k_values[i], resultsMeanAr[i]);
			meanResultList.add(vo);
		}
		
		Collections.sort(meanResultList);
		return meanResultList;
	}
	
	/**
	 * Perform 20%- 80% cross validation and return Mean result of CV.
	 */
	private static double[] performCrossValidation()
	{
		int fifth = (int)Math.floor(NUM_EXAMPLES/5);

		//Perform 20%-80% cross validation
		System.out.println("Break down of data for cross validation...");
		System.out.println(0 + ", " + (fifth -1));
		System.out.println(fifth+ ", " + (fifth*2 -1));
		System.out.println(fifth*2+ ", " + (fifth*3 -1));
		System.out.println(fifth*3+ ", " + (fifth*4 -1));
		System.out.println(fifth*4+ ", " + (NUM_EXAMPLES -1));

		//Each array contains the number of correctly classified test points where
		//the array index corresponds to the k value used
		double[] resultAr1 = _classifyTestData(0, fifth -1);
		double[] resultAr2 = _classifyTestData(fifth, fifth*2 -1);
		double[] resultAr3 = _classifyTestData(fifth*2, fifth*3 -1);
		double[] resultAr4 = _classifyTestData(fifth*3, fifth*4 -1);
		double[] resultAr5 = _classifyTestData(fifth*4, NUM_EXAMPLES -1);
		
		double[] resultsMeanAr = new double[k_values.length];
		
		//Taking mean for the combinations of cross validation
		for(int i=0; i< resultsMeanAr.length; i++)
		{
			resultsMeanAr[i] = (double)(resultAr1[i] + resultAr2[i] + 
								resultAr3[i] + resultAr4[i] + resultAr5[i]) /5;
		}
			
		return resultsMeanAr;
	}
	
	/**
	 * Helper function that partitions the data to train and test and calls
	 * the method to classify test points.
	 */
	public static double[] _classifyTestData(int stIndex, int endIndex)
	{
		int numTestData = (endIndex-stIndex) +1;
		Feature[][] testData = new Feature[numTestData][];
		Feature[][] trainData = new Feature[NUM_EXAMPLES-numTestData][];

		int m = 0;
		int n = 0;
		for(int i=0; i<trainDataAr.length; i++)
		{
			if(i <= endIndex && i >= stIndex)
			{
				testData[m] = trainDataAr[i];
				m++;
			}
			else
			{
				trainData[n] = trainDataAr[i];
				n++;
			}
		}
		
		return classifyTestData(trainData, testData);
	}
	
	
	/**
	 * Classify the test data and return the results for each K
	 */
	public static double[] classifyTestData(Feature[][] trainData, Feature[][] testData)
	{
		double[] percentCorrectAr = new double[k_values.length];
		
		//For each K value
		for(int k = 0; k < k_values.length; k++)
		{
			int numCorrect = 0;

			//For all test points
			for(int i=0; i< testData.length; i++)
			{
				//long time = System.currentTimeMillis();
				List<DistLabelTuple> voList = new ArrayList<DistLabelTuple>();

				//For all training points
				for(int j =0; j < trainData.length; j++)
				{
					double eucDist = calcEuclideanDist(testData[i], trainData[j]);
					DistLabelTuple vo = new DistLabelTuple(eucDist, trainData[j][0].value);
					voList.add(vo);
				}

				int knnTestLabel = determineTestLabel(voList, k_values[k]);
				
				int actual = (int)(testData[i][0].value);
				
				if(actual == knnTestLabel)
				{
					if(actual == EMOTION_ABSENT)
					{
						tp_Absent++;
					}
					else if(actual == EMOTION_NEGATIVE)
						tp_Negative++;
					else if(actual == EMOTION_POSITIVE)
						tp_Positive++;
					else
						tp_Uncertain++;
					
					numCorrect ++;
				}
				else
				{
					if(knnTestLabel == EMOTION_ABSENT)
					{
						fp_Absent++;
					}
					else if(knnTestLabel == EMOTION_NEGATIVE)
						fp_Negative++;
					else if(knnTestLabel == EMOTION_POSITIVE)
						fp_Positive++;
					else
						fp_Uncertain++;

				}
			}
			percentCorrectAr[k] = (double)numCorrect/testData.length;
		}
		
		return percentCorrectAr;
	}
	
	/**
	 * Calculate the Euclidean distance between a test point and a training point
	 */
	private static double calcEuclideanDist(Feature[] testPt, Feature[] trainPt)
	{
		double eucDist = 0.0;
		
		int i = 1;
		int j = 1;
		
		while(i < testPt.length || j < trainPt.length)
		{
			if(i >= testPt.length)
			{
				eucDist += Math.pow(trainPt[j].value, 2);
				j++;
				continue;
			}
			if(j >= trainPt.length)
			{
				eucDist += Math.pow(testPt[i].value, 2);
				i++;
				continue;
			}
			
			if(testPt[i].index == trainPt[j].index)
			{
				eucDist += Math.pow((testPt[i].value - trainPt[j].value), 2);
				i++;
				j++;
			}
			else if(testPt[i].index < trainPt[j].index)
			{
				eucDist += Math.pow(testPt[i].value, 2);
				i++;
			}
			else
			{
				eucDist += Math.pow(trainPt[j].value, 2);
				j++;
			}	
		}
		return eucDist;		
	}
	
	/**
	 * This method takes a list of knnValueObjects each of which contain
	 * Euclidean distances. They are then sorted by distance and the K least
	 * distances are chosen and a majority voting is taken to classify the test
	 * point
	 */
	private static int determineTestLabel(List<DistLabelTuple> voList, int k)
	{
		Collections.sort(voList);

		int[] freqArray = new int[4];
		
		for(int i=0; i < k; i++)
		{
			double trainingNum = (voList.get(i).getTrainingNumeral());
			freqArray[(int)trainingNum]++;
		}
	
		int maxFreq = -1;
		int maxFreqIndex = -1;
		
		for(int i=0; i< freqArray.length; i++)
		{
			if(freqArray[i] > maxFreq)
			{
				maxFreq = freqArray[i];
				maxFreqIndex = i;
			}
		}
		return maxFreqIndex;
	}
	
	public static void readDataFromFile(String fileName)
	{		
		BufferedReader br = null;
		try
		{
			br = new BufferedReader(new FileReader(fileName));
			String line = null;
			
			int j = 0;
			while ((line = br.readLine()) != null) 
	        {
	        	String[] attributes = line.split(" ");
	        	List<Feature> trainPt = new ArrayList<Feature>();
				
				boolean emotionPresent = (Double.parseDouble(attributes[0])==1.0);
				boolean emotionPos = (Double.parseDouble(attributes[1])==1.0);
				boolean emotionNeg = (Double.parseDouble(attributes[1])== -1.0);

				if(emotionPresent)
				{
					if(emotionPos)
						trainPt.add(new Feature(0, EMOTION_POSITIVE));
					else if(emotionNeg)
						trainPt.add(new Feature(0, EMOTION_NEGATIVE));
					else
						trainPt.add(new Feature(0, EMOTION_UNCERTAIN));
				}
				else
				{
					trainPt.add(new Feature(0, EMOTION_ABSENT));
				}
					
				for(int i=2; i< attributes.length; i++)
				{
					double value = Double.parseDouble(attributes[i]);
					if(value != 0.0)
						trainPt.add(new Feature(i-1, value));
				}
	
				trainDataAr[j] = trainPt.toArray(new Feature[1]);
				j++;
	        }
		}
		catch(IOException e)
		{
			System.err.println(e.getMessage());
		}
		finally
		{
			if(br != null)
			{
				try
				{
					br.close();
				}
				catch(IOException e)
				{
					System.err.println(e.getMessage());
				}
			}
		}
	}
	
	/**
	 * Figure out dimensions before processing so we can deal with arrays.
	 */
	public static void calculateDimensions(String FILE)
	{
		int numExamples=0;
		
		BufferedReader br = null;
		try
		{
			br = new BufferedReader(new FileReader(FILE));
			
			String line = null;
			while ((line = br.readLine()) != null) 
	        {
				if(numExamples == 0)
				{
					String[] attributes = line.split(" ");
					//-2 because ignore emotion_present and emotion_valence values
					NUM_ATTRIBUTES = attributes.length - 2; 
				}
				numExamples++;
	        }
			NUM_EXAMPLES = numExamples;
		}
		catch(IOException e)
		{
			System.err.println(e.getMessage());
		}
		finally
		{
			if(br != null)
			{
				try
				{
					br.close();
				}
				catch(IOException e)
				{
					System.err.println(e.getMessage());
				}
			}
		}
	}
	
	/**
	 * Value object: A tuple that contains Euclidean distance from a test point
	 * to a training point and the label of that training point.
	 */
	private static class DistLabelTuple implements Comparable<DistLabelTuple>
	{
		private double eucDist;
		private double trainLabel;
		
		public DistLabelTuple(double eucDist, double trainLabel)
		{
			this.eucDist = eucDist;
			this.trainLabel = trainLabel;
		}
		
		public int compareTo(DistLabelTuple vo)
		{
			if(this.eucDist - vo.eucDist > 0)
				return 1;
			else if(this.eucDist - vo.eucDist < 0)
				return -1;
			else
				return 0;
		}
		
		public double getEuclideanDist()
		{
			return eucDist;
		}
		
		public double getTrainingNumeral()
		{
			return trainLabel;
		}
	}
	
	private static class KvalAccuracyTuple implements Comparable<KvalAccuracyTuple>
	{
		private int k;
		private double accuracy;
		
		public KvalAccuracyTuple(int k, double accuracy)
		{
			this.k = k;
			this.accuracy = accuracy;
		}
		
		public int compareTo(KvalAccuracyTuple vo)
		{
			if(this.accuracy - vo.accuracy > 0)
				return -1;
			else if(this.accuracy - vo.accuracy < 0)
				return +1;
			else
				return 0;
		}
		
		public double getAccuracy()
		{
			return accuracy;
		}
		
		public int getK()
		{
			return k;
		}
	}
	
	public static void printFeatures()
	{
		for(int i=0; i < trainDataAr.length; i++)
		{
			for(int j=0; j < trainDataAr[i].length; j++)
			{
				System.out.print(trainDataAr[i][j].index + ":" + trainDataAr[i][j].value + " ");
			}
			System.out.println("");
		}
	}
}