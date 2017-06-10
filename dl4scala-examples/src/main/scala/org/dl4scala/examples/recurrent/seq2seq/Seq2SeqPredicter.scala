package org.dl4scala.examples.recurrent.seq2seq

import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

/**
  * Created by endy on 2017/6/10.
  */
class Seq2SeqPredicter(net: ComputationGraph) {

  private var decoderInputTemplate: INDArray = _

  def output(testSet: MultiDataSet): INDArray =
    if (testSet.getFeatures(0).size(0) > 2) output(testSet, print = false) else output(testSet, print = true)

  def output(testSet: MultiDataSet, print: Boolean): INDArray = {
    val correctOutput = testSet.getLabels(0)
    var ret = Nd4j.zeros(correctOutput.shape(), DataBuffer.Type.INT)
    decoderInputTemplate = testSet.getFeatures(1).dup

    var currentStepThrough = 0
    val stepThroughs = correctOutput.size(2) - 1

    while (currentStepThrough < stepThroughs) {
      if (print) {
        System.out.println("In time step " + currentStepThrough)
        System.out.println("\tEncoder input and Decoder input:")
        System.out.println(CustomSequenceIterator.mapToString(testSet.getFeatures(0), decoderInputTemplate, " +  "))
      }
      ret = stepOnce(testSet, currentStepThrough)
      if (print) {
        System.out.println("\tDecoder output:")
        System.out.println("\t" + CustomSequenceIterator.oneHotDecode(ret).mkString("\n\t"))
      }
      currentStepThrough += 1
    }
  }



  // Will do a forward pass through encoder + decoder with the given input
  // Updates the decoder input template from time = 1 to time t=n+1;
  // Returns the output from this forward pass
  private def stepOnce(testSet: MultiDataSet, n: Int): INDArray = {
    val currentOutput = net.output(false, testSet.getFeatures(0), decoderInputTemplate)(0)
    copyTimeSteps(n, currentOutput, decoderInputTemplate)
    currentOutput
  }

  // Copies timesteps
  // time = 0 to time = t in "fromArr"
  // to time = 1 to time = t+1 in "toArr"
  private def copyTimeSteps(t: Int, fromArr: INDArray, toArr: INDArray) = {
    val fromView = fromArr.get(NDArrayIndex.all, NDArrayIndex.all, NDArrayIndex.interval(0, t, true))
    val toView = toArr.get(NDArrayIndex.all, NDArrayIndex.all, NDArrayIndex.interval(1, t + 1, true))
    toView.assign(fromView.dup)
  }
}
