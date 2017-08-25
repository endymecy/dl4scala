package org.dl4scala.examples.misc.customlayers.layer

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.gradient.{DefaultGradient, Gradient}
import org.deeplearning4j.nn.layers.BaseLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.primitives



/**
  * Created by endy on 2017/7/1.
  */
class CustomLayerImpl(conf: NeuralNetConfiguration) extends BaseLayer[CustomLayer](conf) {
  override def isPretrainLayer: Boolean = false

  override def preOutput(x: INDArray, training: Boolean): INDArray = {
    super.preOutput(x, training)
  }

  override def activate(training: Boolean): INDArray = {
    val output = preOutput(training)
    val columns = output.columns

    val firstHalf = output.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val secondHalf = output.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val activation1 = layerConf().getActivationFn
    val activation2 = conf.getLayer.asInstanceOf[CustomLayer].getSecondActivationFunction

    // IActivation function instances modify the activation functions in-place
    activation1.getActivation(firstHalf, training)
    activation2.getActivation(secondHalf, training)
    output
  }

  override def backpropGradient(epsilon: INDArray): primitives.Pair[Gradient, INDArray] = {
    val activationDerivative = preOutput(true)
    val columns = activationDerivative.columns

    val firstHalf = activationDerivative.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val secondHalf = activationDerivative.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val epsilonFirstHalf = epsilon.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val epsilonSecondHalf = epsilon.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val activation1 = layerConf().getActivationFn
    val activation2 = conf.getLayer.asInstanceOf[CustomLayer].getSecondActivationFunction

    //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
    activation1.backprop(firstHalf, epsilonFirstHalf)
    activation2.backprop(secondHalf, epsilonSecondHalf)


    if (maskArray != null) activationDerivative.muliColumnVector(maskArray)

    val ret = new DefaultGradient

    val weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY) //f order
    Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0)
    val biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY)
    biasGrad.assign(activationDerivative.sum(0)) //TODO: do this without the assign


    ret.gradientForVariable.put(DefaultParamInitializer.WEIGHT_KEY, weightGrad)
    ret.gradientForVariable.put(DefaultParamInitializer.BIAS_KEY, biasGrad)

    val epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose).transpose

    new primitives.Pair(ret, epsilonNext)
  }
}
