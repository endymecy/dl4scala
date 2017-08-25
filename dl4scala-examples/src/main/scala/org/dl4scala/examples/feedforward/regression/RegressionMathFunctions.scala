package org.dl4scala.examples.feedforward.regression

import javax.swing.{JFrame, WindowConstants}

import scala.util.Random
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.dl4scala.examples.feedforward.regression.function.{MathFunction, SinXDivXMathFunction}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs

import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/5/25.
  */

object RegressionMathFunctions extends App{
  // Random number generator seed, for reproducability

  val seed = 12345
  // Number of iterations per mini batch
  val iterations = 1
  // Number of epochs (full passes of the data)
  val nEpochs = 2000
  // How frequently should we plot the network output?
  val plotFrequency = 500
  // Number of data points
  val nSamples = 1000
  // Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  val batchSize = 100
  // Network learning rate
  val learningRate = 0.01
  val rng = new Random(seed)
  val numInputs = 1
  val numOutputs = 1

  private def getDeepDenseLayerNetworkConfiguration: MultiLayerConfiguration = {
    val numHiddenNodes = 50
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(new Nesterovs(0.9))
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation(Activation.TANH).build())
      .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        .activation(Activation.TANH).build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(numHiddenNodes).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build()
  }

  private def getTrainingData(x: INDArray, function: MathFunction, batchSize: Int, rng: Random) = {
    val y = function.getFunctionValues(x)
    val allData = new DataSet(x, y)
    val list = allData.asList
    val shuffle = Random.shuffle(list.asScala)
    new ListDataSetIterator(shuffle.asJavaCollection, batchSize)
  }

  // Plot the data
  private def plot(function: MathFunction, x: INDArray, y: INDArray, predicted: Array[INDArray]) = {
    val dataSet = new XYSeriesCollection
    addSeries(dataSet, x, y, "True Function (Labels)")

    for(i <- predicted.indices){
      addSeries(dataSet, x, predicted(i), String.valueOf(i))
    }


    val chart = ChartFactory.createXYLineChart("Regression Example - " + function.getName, // chart title
      "X", // x axis label
      function.getName + "(X)", // y axis label
      dataSet, // data
      PlotOrientation.VERTICAL, true, // include legend
      true, // tooltips
      false // urls
     )

    val panel: ChartPanel = new ChartPanel(chart)

    val f: JFrame = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()

    f.setVisible(true)
  }

  private def addSeries(dataSet: XYSeriesCollection, x: INDArray, y: INDArray, label: String) = {
    val xd = x.data.asDouble
    val yd = y.data.asDouble
    val s = new XYSeries(label)
    for(j <- xd.indices) s.add(xd(j), yd(j))

    dataSet.addSeries(s)
  }

  // Switch these two options to do different functions with different networks
  val fn = new SinXDivXMathFunction()
  val conf = getDeepDenseLayerNetworkConfiguration

  // Generate the training data
  val x = Nd4j.linspace(-10, 10, nSamples).reshape(nSamples, 1)
  val iterator = getTrainingData(x, fn, batchSize, rng)

  // Create the network
  val net = new MultiLayerNetwork(conf)
  net.init()
  net.setListeners(new ScoreIterationListener(1))

  // Train the network on the full data set, and evaluate in periodically
  val networkPredictions = new Array[INDArray](nEpochs / plotFrequency)

  (0 until nEpochs).foreach{i =>
    iterator.reset()
    net.fit(iterator)
    if ((i + 1) % plotFrequency == 0) networkPredictions(i / plotFrequency) = net.output(x, false)
  }
  // Plot the target data and the network predictions
  plot(fn, x, fn.getFunctionValues(x), networkPredictions)
}
