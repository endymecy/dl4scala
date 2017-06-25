package org.dl4scala.examples.userInterface

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.dl4scala.examples.userInterface.util.{GradientsAndParamsListener, UIExampleUtils}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

/**
  * Created by endy on 2017/6/25.
  */
object GradientsAndParamsListenerExample {
  def main(args: Array[String]): Unit = {

    // Get our network and training data
    val net: MultiLayerNetwork = UIExampleUtils.getMnistNetwork
    val trainData: DataSetIterator = UIExampleUtils.getMnistData

    System.out.println()
    for (layer <- net.getLayers) {
      System.out.println(layer)
    }
    System.out.println()
    net.setListeners(new GradientsAndParamsListener(net, 100))
  }
}
