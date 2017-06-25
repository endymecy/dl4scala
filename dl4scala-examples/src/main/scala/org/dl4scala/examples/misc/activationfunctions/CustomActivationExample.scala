package org.dl4scala.examples.misc.activationfunctions

import java.util.{Collections, Random}

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * This is an example that illustrates how to instantiate and use a custom activation function.
  *
  * Created by endy on 2017/6/25.
  */
object CustomActivationExample {
  val seed = 12345
  val iterations = 1
  val nEpochs = 500
  val nSamples = 1000
  val batchSize = 100
  val learningRate = 0.001
  var MIN_RANGE = 0
  var MAX_RANGE = 3

  val rng = new Random(seed)

  def main(args: Array[String]): Unit = {
    val iterator = getTrainingData(batchSize, rng)

    // Create the network
    val numInput = 2
    val numOutputs = 1
    val nHidden = 10

    val net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS).momentum(0.95)
      .list()
      //INSTANTIATING CUSTOM ACTIVATION FUNCTION here as follows
      //Refer to CustomActivation class for more details on implementation
      .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
      .activation(new CustomActivation())
      .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(nHidden).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build()
    )

    net.init()
    net.setListeners(new ScoreIterationListener(100))

    (0 until nEpochs).foreach{_ =>
      iterator.reset()
      net.fit(iterator)
    }

    // Test the addition of 2 numbers (Try different numbers here)
    val input: INDArray = Nd4j.create(Array[Double](0.111111, 0.3333333333333), Array[Int](1, 2))
    val out: INDArray = net.output(input, false)
    System.out.println(out)
  }


  private def getTrainingData(batchSize: Int, rand: Random): DataSetIterator = {
    val sum = new Array[Double](nSamples)
    val input1 = new Array[Double](nSamples)
    val input2 = new Array[Double](nSamples)

    (0 until nSamples).foreach{i =>
      input1(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble
      input2(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble
      sum(i) = input1(i) + input2(i)
    }

    val inputNDArray1 = Nd4j.create(input1, Array[Int](nSamples, 1))
    val inputNDArray2 = Nd4j.create(input2, Array[Int](nSamples, 1))
    val inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2)
    val outPut = Nd4j.create(sum, Array[Int](nSamples, 1))
    val dataSet = new DataSet(inputNDArray, outPut)
    val listDs = dataSet.asList
    Collections.shuffle(listDs, rng)
    new ListDataSetIterator(listDs, batchSize)
  }
}
