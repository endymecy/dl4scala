package org.dl4scala.examples.feedforward.anomalydetection

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import java.util.{Collections, Random}

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.optimize.api.IterationListener

import scala.collection.mutable.ArrayBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable

/**
  * Example: Anomaly Detection on MNIST using simple autoencoder without pretraining
  * The goal is to identify outliers digits, i.e., those digits that are unusual or
  * not like the typical digits.
  * This is accomplished in this example by using reconstruction error: stereotypical
  * examples should have low reconstruction error, whereas outliers should have high
  * reconstruction error
  * Created by endy on 2017/5/24.
  */
object MNISTAnomalyExample extends App{
  val logger: Logger = LoggerFactory.getLogger(MNISTAnomalyExample.getClass)

  implicit def ordered: Ordering[(Double, INDArray)] = new Ordering[(Double, INDArray)] {
    def compare(x: (Double, INDArray), y: (Double, INDArray)): Int = x._1 compareTo y._1
  }

  // Set up network. 784 in/out (as MNIST images are 28x28).
  // 784 -> 250 -> 10 -> 250 -> 784
  val conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .iterations(1)
    .weightInit(WeightInit.XAVIER)
    .updater(Updater.ADAGRAD)
    .activation(Activation.RELU)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(0.05)
    .regularization(true).l2(0.0001)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
      .build())
    .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
      .build())
    .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
      .build())
    .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
      .lossFunction(LossFunctions.LossFunction.MSE)
      .build())
    .pretrain(false).backprop(true)
    .build()


  val net = new MultiLayerNetwork(conf)
  net.setListeners(Collections.singletonList(new ScoreIterationListener(1).asInstanceOf[IterationListener]))

  // Load data and split into training and testing sets. 40000 train, 10000 test
  val iter = new MnistDataSetIterator(100, 50000, false)

  val featuresTrain = new ArrayBuffer[INDArray]
  val featuresTest = new ArrayBuffer[INDArray]
  val labelsTest = new ArrayBuffer[INDArray]

  val r: Random = new Random(12345)

  while(iter.hasNext){
    val ds: DataSet = iter.next
    val split: SplitTestAndTrain = ds.splitTestAndTrain(80, r) // 80/20 split (from miniBatch = 100)
    featuresTrain.append(split.getTrain.getFeatureMatrix)
    val dsTest: DataSet = split.getTest
    featuresTest.append(dsTest.getFeatureMatrix)
    val indexes: INDArray = Nd4j.argMax(dsTest.getLabels, 1) // Convert from one-hot representation -> index
    labelsTest.append(indexes)
  }

  // train model
  for (epoch <- 0 until 30){
    for (data <- featuresTrain) {
      net.fit(data, data)
    }
    logger.info("Epoch " + epoch + " complete")
  }

  // Evaluate the model on the test data
  // Score each example in the test set separately
  // Compose a map that relates each digit to a list of (score, example) pairs
  // Then find N best and N worst scores per digit

  var listsByDigit = new mutable.OpenHashMap[Int, ArrayBuffer[(Double, INDArray)]]
  (0 until 10).foreach(i => listsByDigit.put(i, new ArrayBuffer[(Double, INDArray)]()))

  for(i <- featuresTest.indices){
    val testData = featuresTest(i)
    val labels = labelsTest(i)
    val nRows = testData.rows
    for(j <- 0 until nRows){
      val example = testData.getRow(j)
      val digit = labels.getDouble(j).asInstanceOf[Int]
      val score = net.score(new DataSet(example, example))
      // Add (score, example) pair to the appropriate list
      val digitAllPairs: ArrayBuffer[(Double, INDArray)] = listsByDigit(digit)
      digitAllPairs.append((score, example))
    }
  }

  val sortedListByDigit = new mutable.OpenHashMap[Int, ArrayBuffer[(Double, INDArray)]]

  for(key <- listsByDigit.keys){
    val oneDigit = listsByDigit(key).sorted
    sortedListByDigit.put(key, oneDigit)
  }

  // After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
  val best = new ArrayBuffer[INDArray](100)
  val worst = new ArrayBuffer[INDArray](100)

  (0 until 10).foreach { i =>
    val list = sortedListByDigit(i)
    (0 until 10).foreach{j =>
      worst.append(list(j)._2)
      best.append(list(list.size - j - 1)._2)
    }
  }

  // Visualize the best and worst digits//Visualize the best and worst digits
  val bestVisualizer = new MNISTVisualizer(2.0, best, "Best (Low Rec. Error)", 10)
  bestVisualizer.visualize()

  val worstVisualizer = new MNISTVisualizer(2.0, worst, "Worst (High Rec. Error)", 10)
  worstVisualizer.visualize()
}


