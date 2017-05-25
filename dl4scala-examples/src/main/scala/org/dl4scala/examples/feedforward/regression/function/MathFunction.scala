package org.dl4scala.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by endy on 2017/5/25.
  */
trait MathFunction {
  def getFunctionValues(x: INDArray): INDArray

  def getName: String
}
