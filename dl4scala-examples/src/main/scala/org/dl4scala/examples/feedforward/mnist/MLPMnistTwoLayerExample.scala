package org.dl4scala.examples.feedforward.mnist

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.examples.feedforward.mnist.MLPMnistTwoLayerExample
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

object MLPMnistTwoLayerExample {

  private val log: Logger = LoggerFactory.getLogger(classOf[MLPMnistTwoLayerExample])

  def main(args: scala.Array[String]): Unit = {
    //number of rows and columns in the input pictures
    val numRows: Int = 28
    val numColumns: Int = 28
    val outputNum: Int = 10 // number of output classes
    val batchSize: Int = 64 // batch size for each epoch
    val rngSeed: Int = 123 // random number seed for reproducibility
    val numEpochs: Int = 1 // number of epochs to perform
    val rate: Double = 0.0015 // learning rate

    //Get the DataSetIterators:
    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, rngSeed)

    log.info("Build model....")

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
        .seed(rngSeed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(1)
        .activation("relu")
        .weightInit(WeightInit.XAVIER)
        .learningRate(rate) //specify the learning rate
        .updater(Updater.NESTEROVS).momentum(0.98) //specify the rate of change of the learning rate.
        .regularization(true).l2(rate * 0.005) // regularize learning model
        .list
        .layer(0, new DenseLayer.Builder().nIn(numRows * numColumns).nOut(500).build)
        .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build)
        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(100).nOut(outputNum).build)
        .pretrain(false).backprop(true)
        .build

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(5)) //print the score with every iteration
    log.info("Train model....")

    (0 until numEpochs).foreach { i =>
      log.info("Epoch " + i)
      model.fit(mnistTrain)
    }

    log.info("Evaluate model....")
    val eval: Evaluation = new Evaluation(outputNum) //create an evaluation object with 10 possible classes
    while (mnistTest.hasNext) {
      val next: DataSet = mnistTest.next
      val output: INDArray = model.output(next.getFeatureMatrix) //get the networks prediction
      eval.eval(next.getLabels, output) //check the prediction against the true class
    }
    log.info(eval.stats)
    log.info("****************Example finished********************")
  }
}