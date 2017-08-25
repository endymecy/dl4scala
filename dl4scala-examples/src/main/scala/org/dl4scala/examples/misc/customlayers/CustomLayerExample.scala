package org.dl4scala.examples.misc.customlayers

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import java.io.{File, IOException}

import org.deeplearning4j.gradientcheck.GradientCheckUtil
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.{BaseLayer, DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.dl4scala.examples.misc.customlayers.layer.CustomLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.util.Random

/**
  * Created by endy on 2017/7/1.
  */
object CustomLayerExample {
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  def main(args: Array[String]): Unit = {
    runInitialTests()
    doGradientCheck()
  }

  @throws(classOf[IOException])
  private def runInitialTests() = {
    System.out.println("----- Starting Initial Tests -----")
    val nIn = 5
    val nOut = 8

    val config = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .updater( new RmsProp(0.95))
      .weightInit(WeightInit.XAVIER)
      .regularization(true).l2(0.03)
      .list()
      .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //Standard DenseLayer
      .layer(1, new CustomLayer.Builder()
             .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
             .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
             .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
             .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
            .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
      .pretrain(false).backprop(true).build()

    //First:  run some basic sanity checks on the configuration:
    val customLayerL2 = config.getConf(1).getLayer.asInstanceOf[BaseLayer].getL2
    System.out.println("l2 coefficient for custom layer: " + customLayerL2) //As expected: custom layer inherits the global L2 parameter configuration

    val customLayerUpdater = config.getConf(1).getLayer.asInstanceOf[BaseLayer].getIUpdater
    System.out.println("Updater for custom layer: " + customLayerUpdater) //As expected: custom layer inherits the global Updater configuration

    //Second: We need to ensure that that the JSON and YAML configuration works, with the custom layer

    // If there were problems with serialization, you'd get an exception during deserialization ("No suitable constructor found..." for example)
    val configAsJson = config.toJson
    val configAsYaml = config.toYaml
    val fromJson = MultiLayerConfiguration.fromJson(configAsJson)
    val fromYaml = MultiLayerConfiguration.fromYaml(configAsYaml)

    System.out.println("JSON configuration works: " + config.equals(fromJson))
    System.out.println("YAML configuration works: " + config.equals(fromYaml))

    val net = new MultiLayerNetwork(config)
    net.init()


    //Third: Let's run some more basic tests. First, check that the forward and backward pass methods don't throw any exceptions
    // To do this: we'll create some simple test data
    val minibatchSize = 5
    val testFeatures = Nd4j.rand(minibatchSize, nIn)
    val testLabels = Nd4j.zeros(minibatchSize, nOut)
    val r = new Random(12345)

    (0 until minibatchSize).foreach(i => testLabels.putScalar(i, r.nextInt(nOut), 1))

    val activations = net.feedForward(testFeatures)
    val activationsCustomLayer = activations.get(2) //Activations index 2: index 0 is input, index 1 is first layer, etc.
    System.out.println("\nActivations from custom layer:")
    System.out.println(activationsCustomLayer)
    net.fit(new DataSet(testFeatures, testLabels))

    //Finally, let's check the model serialization process, using ModelSerializer:
    ModelSerializer.writeModel(net, new File("CustomLayerModel.zip"), true)
    val restored = ModelSerializer.restoreMultiLayerNetwork(new File("CustomLayerModel.zip"))

    System.out.println()
    System.out.println("Original and restored networks: configs are equal: " +
      net.getLayerWiseConfigurations.equals(restored.getLayerWiseConfigurations))
    System.out.println("Original and restored networks: parameters are equal: " +
      net.params.equals(restored.params))
  }

  private def doGradientCheck() = {
    import org.nd4j.linalg.factory.Nd4j
    System.out.println("\n\n\n----- Starting Gradient Check -----")

    Nd4j.getRandom.setSeed(12345)
    val nIn = 3
    val nOut = 2

    val config = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .updater(Updater.NONE).learningRate(1.0)
      .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))              //Larger weight init than normal can help with gradient checks
      .regularization(true).l2(0.03)
      .list()
      .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(3).build())    //Standard DenseLayer
      .layer(1, new CustomLayer.Builder()
      .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
      .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
      .nIn(3).nOut(3)                                                                 //nIn and nOut also inherited from FeedForwardLayer
      .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
        .activation(Activation.SOFTMAX).nIn(3).nOut(nOut).build())
      .pretrain(false).backprop(true).build()

    val net = new MultiLayerNetwork(config)
    net.init()

    val print = true
    //Whether to print status for each parameter during testing
    val return_on_first_failure = false
    //If true: terminate test on first failure
    val gradient_check_epsilon = 1e-8
    //Epsilon value used for gradient checks
    val max_relative_error = 1e-5
    //Maximum relative error allowable for each parameter
    val min_absolute_error = 1e-10 //Minimum absolute error, to avoid failures on 0 vs 1e-30, for example.

    //Create some random input data to use in the gradient check
    val minibatchSize = 5
    val features = Nd4j.rand(minibatchSize, nIn)
    val labels = Nd4j.zeros(minibatchSize, nOut)
    val r = new Random(12345)

    (0 until minibatchSize).foreach(i => labels.putScalar(i,r.nextInt(nOut),1))

    (0 until 3).foreach(i => System.out.println("# params, layer " + i + ":\t" + net.getLayer(i).numParams()))
    GradientCheckUtil.checkGradients(net, gradient_check_epsilon, max_relative_error, min_absolute_error, print,
      return_on_first_failure, features, labels)
  }
}
