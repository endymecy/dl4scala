package org.dl4scala.examples.feedforward.mnist

import org.slf4j.LoggerFactory
import org.slf4j.Logger
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Created by endy on 2017/5/13.
  */
object MLPMnistSingleLayerExample extends App{
  private val log: Logger = LoggerFactory.getLogger(MLPMnistSingleLayerExample.getClass)

  // number of rows and columns in the input pictures//number of rows and columns in the input pictures
  private  val numRows = 28
  private  val numColumns = 28
  private  val outputNum = 10 // number of output classes
  private  val batchSize = 128 // batch size for each epoch
  private  val rngSeed = 123 // random number seed for reproducibility
  private  val numEpochs = 15 // number of epochs to perform

  // Get the DataSetIterators://Get the DataSetIterators:
  val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
  val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)


  log.info("Build model....")

  val conf = new NeuralNetConfiguration
    .Builder()
    .seed(rngSeed) // include a random seed for reproducibility
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
    .iterations(1)
    .learningRate(0.006) // specify the learning rate
    .updater(Updater.NESTEROVS).momentum(0.9) // specify the rate of change of the learning rate.
    .regularization(true).l2(1e-4)
    .list()
    .layer(0, new DenseLayer.Builder() // create the first, input layer with xavier initialization
        .nIn(numRows * numColumns)
        .nOut(1000)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
        .nIn(1000)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build())
    .pretrain(false).backprop(true) //use backpropagation to adjust weights
    .build()

  val model = new MultiLayerNetwork(conf)
  model.init()
  // print the score with every 1 iteration
  model.setListeners(new ScoreIterationListener(1))
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
