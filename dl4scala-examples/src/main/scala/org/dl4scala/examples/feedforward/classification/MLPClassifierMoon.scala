package org.dl4scala.examples.feedforward.classification

import java.io.File

import org.datavec.api.util.ClassPathResource
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by endy on 2017/5/13.
  */
object MLPClassifierMoon extends App{
  private val log: Logger = LoggerFactory.getLogger(MLPClassifierMoon.getClass)

  val seed = 123
  val learningRate = 0.005
  val batchSize = 50
  val nEpochs = 100

  val numInputs = 2
  val numOutputs = 2
  val numHiddenNodes = 20

  val filenameTrain = new ClassPathResource("/classification/moon_data_train.csv").getFile.getPath
  val filenameTest = new ClassPathResource("/classification/moon_data_eval.csv").getFile.getPath

  // Load the training data:
  val rr = new CSVRecordReader
  rr.initialize(new FileSplit(new File(filenameTrain)))
  var trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2)

  // Load the test/evaluation data://Load the test/evaluation data:
  val rrTest = new CSVRecordReader
  rrTest.initialize(new FileSplit(new File(filenameTest)))
  var testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2)

  log.info("Build model....")
  val conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(1)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .updater(new Nesterovs(0.9))
    .list()
    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
      .weightInit(WeightInit.XAVIER)
      .activation(Activation.RELU)
      .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
      .weightInit(WeightInit.XAVIER)
      .activation(Activation.SOFTMAX)
      .nIn(numHiddenNodes).nOut(numOutputs).build())
    .pretrain(false).backprop(true)
    .build()

  val model = new MultiLayerNetwork(conf)
  model.init()
  model.setListeners(new ScoreIterationListener(100)) // Print score every 10 parameter updates

  for(i <- 0 until nEpochs){
    model.fit(trainIter)
  }

  log.info("Evaluate model....")
  val eval = new Evaluation(numOutputs)
  while (testIter.hasNext){
    val t = testIter.next
    val predicted = model.output(t.getFeatureMatrix, false)
    eval.eval(t.getLabels, predicted)
  }
  log.info(eval.stats())

  // ------------------------------------------------------------------------------------
  // Training is complete. Code that follows is for plotting the data & predictions only

  // Plot the data
  val xMin = -1.5
  val xMax = 2.5
  val yMin = -1
  val yMax = 1.5

  // Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
  val nPointsPerAxis = 100

  val evalPoints = Array.ofDim[Double](nPointsPerAxis * nPointsPerAxis, 2)

  var count = 0
  for(i <- 0 until nPointsPerAxis)
    for(j <- 0 until nPointsPerAxis){
      val x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin
      val y = i * (yMax - yMin) / (nPointsPerAxis - 1) + yMin
      evalPoints(count)(0) = x
      evalPoints(count)(1) = y
      count = count + 1
    }


  val allXYPoints = Nd4j.create(evalPoints)
  val predictionsAtXYPoints = model.output(allXYPoints)

  // Get all of the training data in a single array, and plot it://Get all of the training data in a single array, and plot it:
  rr.initialize(new FileSplit(new ClassPathResource("/classification/moon_data_train.csv").getFile))
  rr.reset()
  val nTrainPoints = 2000
  trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 2)
  var ds = trainIter.next
  PlotUtil.plotTrainingData(ds.getFeatures, ds.getLabels, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

  // Get test data, run the test data through the network to generate predictions, and plot those predictions:
  rrTest.initialize(new FileSplit(new ClassPathResource("/classification/moon_data_eval.csv").getFile))
  rrTest.reset()
  val nTestPoints = 1000
  testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 2)
  ds = testIter.next
  val testPredicted = model.output(ds.getFeatures)
  PlotUtil.plotTestData(ds.getFeatures, ds.getLabels, testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

  log.info("****************Example finished********************")
}
