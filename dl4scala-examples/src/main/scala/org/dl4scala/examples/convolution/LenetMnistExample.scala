package org.dl4scala.examples.convolution

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.LearningRatePolicy
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/5/25.
  */
object LenetMnistExample extends App{
  private val log: Logger = LoggerFactory.getLogger(LenetMnistExample.getClass)

  // implicit def scalaDouble2JavaDouble(d: Double): Double = d: java.lang.Double

  // Number of input channels
  val nChannels = 1
  // The number of possible outcomes
  val outputNum = 10
  // Test batch size
  val batchSize = 64
  // Number of training epochs
  val nEpochs = 10
  // Number of training iterations
  val iterations = 10

  val seed = 123

  // Create an iterator using the batch size for one iteration
  log.info("Load data....")
  val mnistTrain = new MnistDataSetIterator(batchSize, true, 12345)
  val mnistTest = new MnistDataSetIterator(batchSize, false, 12345)

  // Construct the neural network
  log.info("Build model....")

  // learning rate schedule in the form of <Iteration #, Learning Rate>
  val lrSchedule = new mutable.OpenHashMap[Integer, java.lang.Double]()
  lrSchedule.put(0, 0.01)
  lrSchedule.put(1000, 0.005)
  lrSchedule.put(3000, 0.001)

  val conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations) // Training iterations as above
    .regularization(true).l2(0.0005)
    .learningRate(.01) // .biasLearningRate(0.02)
    .learningRateDecayPolicy(LearningRatePolicy.Schedule)
    .learningRateSchedule(lrSchedule.asJava)
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(new Nesterovs(0.9))
    .list()
    .layer(0, new ConvolutionLayer.Builder(5, 5)
      // nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
      .nIn(nChannels)
      .stride(1, 1)
      .nOut(20)
      .activation(Activation.IDENTITY)
      .build())
    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build())
    .layer(2, new ConvolutionLayer.Builder(5, 5)
      // Note that nIn need not be specified in later layers
      .stride(1, 1)
      .nOut(50)
      .activation(Activation.IDENTITY)
      .build())
    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build())
    .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
      .nOut(500).build())
    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .activation(Activation.SOFTMAX)
      .build())
    .setInputType(InputType.convolutionalFlat(28, 28, 1)) // See note below
    .pretrain(false).backprop(false)
    .build()

  /*
       Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
       (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
             and the dense layer
       (b) Does some additional configuration validation
       (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
           layer based on the size of the previous layer (but it won't override values manually set by the user)
       InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
       For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
       MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
       row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
   */

  val model = new MultiLayerNetwork(conf)
  model.init()

  log.info("Train model....")
  model.setListeners(new ScoreIterationListener(1))

  (0 until nEpochs).foreach{i =>
    model.fit(mnistTrain)
    log.info("*** Completed epoch {} ***", i)

    log.info("Evaluate model....")
    val eval = model.evaluate(mnistTest)
    log.info(eval.stats)
    mnistTest.reset()
  }
  log.info("****************Example finished********************")

}
