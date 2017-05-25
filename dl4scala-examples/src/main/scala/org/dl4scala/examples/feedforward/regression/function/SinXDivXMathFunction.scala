package org.dl4scala.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Sin
import org.nd4j.linalg.factory.Nd4j


/**
  * Created by endy on 2017/5/25.
  */
class SinXDivXMathFunction extends MathFunction {

  override def getFunctionValues(x: INDArray): INDArray = {
    Nd4j.getExecutioner.execAndReturn(new Sin(x.dup)).div(x)
  }

  override def getName: String = "SinXDivX"

}
