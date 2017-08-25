package org.dl4scala.examples.misc.customlayers.layer

import java.util

import org.deeplearning4j.nn.conf.layers.{FeedForwardLayer, Layer}
import CustomLayer.Builder
import org.deeplearning4j.nn.api
import org.deeplearning4j.nn.api.ParamInitializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.memory.{LayerMemoryReport, MemoryReport}
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.activations.{Activation, IActivation}
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by endy on 2017/7/1.
  */
class CustomLayer(builder: Builder) extends FeedForwardLayer(builder){

  def this()= this(new Builder)

  private var secondActivationFunction = builder.secondActivationFunction

  def getSecondActivationFunction: IActivation =  secondActivationFunction

  def setSecondActivationFunction(secondActivationFunction: IActivation): Unit = { //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
    this.secondActivationFunction = secondActivationFunction
  }

  override def instantiate(conf: NeuralNetConfiguration, iterationListeners: util.Collection[IterationListener],
                           layerIndex: Int, layerParamsView: INDArray, initializeParams: Boolean): api.Layer = {
    //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class

    // (i.e., a CustomLayerImpl instance)
    //For the most part, it's the same for each type of layer

    val myCustomLayer = new CustomLayerImpl(conf)
    myCustomLayer.setListeners(iterationListeners) //Set the iteration listeners, if any
    myCustomLayer.setIndex(layerIndex) //Integer index of the layer

    // Parameter view array: the network parameters for the entire network (all layers) are
    // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
    // (i.e., it's a "view" array in that it's a subset of a larger array)
    // This is a row vector, with length equal to the number of parameters in the layer
    myCustomLayer.setParamsViewArray(layerParamsView)

    // Initialize the layer parameters. For example,
    // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
    // are in turn a view of the 'layerParamsView' array.
    val paramTable = initializer().init(conf, layerParamsView, initializeParams)
    myCustomLayer.setParamTable(paramTable)
    myCustomLayer.setConf(conf)
    myCustomLayer
  }

  // This method returns the parameter initializer for this type of layer
  // In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
  // For more complex layers, you may need to implement a custom parameter initializer
  override def initializer(): ParamInitializer = DefaultParamInitializer.getInstance

  override def getMemoryReport(inputType: InputType): LayerMemoryReport = {
    //Memory report is used to estimate how much memory is required for the layer, for different configurations//Memory report is used to estimate how much memory is required for the layer, for different configurations

    //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
    // with all 0s, or

    //This implementation: based on DenseLayer implementation
    val outputType = getOutputType(-1, inputType)

    val numParams = initializer().numParams(this)
    val updaterStateSize = getIUpdater.stateSize(numParams).asInstanceOf[Int]

    val trainSizeFixed = 0
    var trainSizeVariable = 0
    if (getDropOut > 0) { //Assume we dup the input for dropout
      trainSizeVariable += inputType.arrayElementsPerExample
    }

    //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
    // which is modified in-place by activation function backprop
    // then we have 'epsilonNext' which is equivalent to input size
    trainSizeVariable += outputType.arrayElementsPerExample()

    new LayerMemoryReport.Builder(layerName, CustomLayer.getClass, inputType, outputType)
    .standardMemory(numParams, updaterStateSize)
      .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
      .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
      .build()
  }
}

object CustomLayer {
  class Builder extends FeedForwardLayer.Builder[Builder] {
    var secondActivationFunction: IActivation = _

    /**
      * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
      *
      * @param secondActivationFunction Second activation function for the layer
      */
    def secondActivationFunction(secondActivationFunction: String): Builder =
      this.secondActivationFunction(Activation.fromString(secondActivationFunction))

    /**
      * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
      *
      * @param secondActivationFunction Second activation function for the layer
      */
    def secondActivationFunction(secondActivationFunction: Activation): Builder = {
      this.secondActivationFunction = secondActivationFunction.getActivationFunction
      this
    }

    @unchecked
    override def build[T <: Layer](): T = {
      new CustomLayer(this).asInstanceOf[T]
    }
  }
}