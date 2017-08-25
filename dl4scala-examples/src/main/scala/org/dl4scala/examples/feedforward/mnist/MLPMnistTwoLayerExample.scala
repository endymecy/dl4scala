package org.dl4scala.examples.feedforward.mnist

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Nesterovs

/**
  * Created by endy on 2017/5/13.
  */
object MLPMnistTwoLayerExample extends App{
  private val log: Logger = LoggerFactory.getLogger(MLPMnistTwoLayerExample.getClass)

  // number of rows and columns in the input pictures//number of rows and columns in the input pictures
  private  val numRows = 28
  private  val numColumns = 28
  private  val outputNum = 10 // number of output classes
  private  val batchSize = 128 // batch size for each epoch
  private  val rngSeed = 123 // random number seed for reproducibility
  private  val numEpochs = 15 // number of epochs to perform
  private  val rate: Double = 0.0015 // learning rate

  // Get the DataSetIterators://Get the DataSetIterators:
  private val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
  private val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

  log.info("Build model....")

  private val conf = new NeuralNetConfiguration
    .Builder()
    .seed(rngSeed) // include a random seed for reproducibility
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
    .iterations(1)
    .activation(Activation.RELU)
    .weightInit(WeightInit.XAVIER)
    .learningRate(rate) // specify the learning rate
    .updater(new Nesterovs(0.98))
    .regularization(true).l2(rate * 0.005) // regularize learning model
    .list()
    .layer(0, new DenseLayer.Builder() // create the first input layer.
      .nIn(numRows * numColumns)
      .nOut(500)
      .build())
    .layer(1, new DenseLayer.Builder() // create the second input layer
      .nIn(500)
      .nOut(100)
      .build())
    .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
      .activation(Activation.SOFTMAX)
      .nIn(100)
      .nOut(outputNum)
      .build())
    .pretrain(false).backprop(true)
    .build()

  val model = new MultiLayerNetwork(conf)
  model.init()
  model.setListeners(new ScoreIterationListener(5)) // print the score with every iteration

  log.info("Train model....")

  for(i <- 0 until numEpochs){
    model.fit(mnistTrain)
  }

  log.info("Evaluate model....")
  val eval = new Evaluation(outputNum) // create an evaluation object with 10 possible classes

  while(mnistTest.hasNext){
    val next = mnistTest.next
    val output = model.output(next.getFeatureMatrix) // get the networks prediction
    eval.eval(next.getLabels, output) // check the prediction against the true class
  }

  log.info(eval.stats)
  log.info("****************Example finished********************")
}
