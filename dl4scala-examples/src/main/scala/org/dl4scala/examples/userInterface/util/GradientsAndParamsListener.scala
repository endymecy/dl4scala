package org.dl4scala.examples.userInterface.util

import java.util

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by endy on 2017/6/25.
  */
class GradientsAndParamsListener(network: MultiLayerNetwork, sampleSizePerLayer: Int) extends TrainingListener{

  override def onForwardPass(model: Model, activations: util.List[INDArray]): Unit = ???

  override def onForwardPass(model: Model, activations: util.Map[String, INDArray]): Unit = ???

  override def onEpochEnd(model: Model): Unit = ???

  override def onEpochStart(model: Model): Unit = ???

  override def onGradientCalculation(model: Model): Unit = ???

  override def onBackwardPass(model: Model): Unit = ???

  override def invoke(): Unit = ???

  override def iterationDone(model: Model, iteration: Int): Unit = ???

  override def invoked(): Boolean = ???
}
