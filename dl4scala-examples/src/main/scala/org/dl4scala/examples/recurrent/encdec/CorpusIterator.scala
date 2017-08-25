package org.dl4scala.examples.recurrent.encdec


import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

import scala.collection.mutable.ArrayBuffer

/**
  * Motivation: I want to get asynchronous data iteration while not blocking on net.fit() until the end of epoch. I want to checkpoint
  * the network, show intermediate test results and some stats, it would be harder to achieve with listeners I think so this is how I
  * solved the problem. This way the learn process is asynchronous inside one macrobatch and synchronous across all the macrobatches.
  *
  * Macrobatch is a group of minibatches. The iterator is modified so that it reports the end of data when it exhausts a macrobatch. Then
  * it advances (manually) to the next macrobatch.
  *
  * Created by endy on 2017/6/1.
  */
class CorpusIterator(corpus: ArrayBuffer[ArrayBuffer[Double]], batchSize: Int, batchesPerMacrobatch: Int,
                     dictSize: Int, rowSize: Int) extends MultiDataSetIterator{

  private var currentBatch = 0
  private var currentMacroBatch = 0
  private val totalMacroBatches = Math.ceil(totalBatches().asInstanceOf[Double] / batchesPerMacrobatch).toInt

  override def next(i: Int): MultiDataSet = {
    var i = currentBatch * batchSize
    val currentBatchSize = Math.min(batchSize, corpus.size - i - 1)
    val input = Nd4j.zeros(currentBatchSize, 1, rowSize)
    val prediction = Nd4j.zeros(currentBatchSize, dictSize, rowSize)
    val decode = Nd4j.zeros(currentBatchSize, dictSize, rowSize)
    val inputMask = Nd4j.zeros(currentBatchSize, rowSize)
    // this mask is also used for the decoder input, the length is the same
    val predictionMask = Nd4j.zeros(currentBatchSize, rowSize)

    (0 until currentBatchSize).foreach{j =>
      val old_rowIn = corpus(i)
      val rowIn = old_rowIn.reverse

      val oldPred = corpus(i + 1)
      val rowPred = new ArrayBuffer[Double]()
      rowPred.appendAll(oldPred)
      rowPred.append(1.0) // add <eos> token

      // replace the entire row in "input" using NDArrayIndex, it's faster than putScalar();
      // input is NOT made of one-hot vectors
      // because of the embedding layer that accepts token indexes directly
      input.put(Array[INDArrayIndex](NDArrayIndex.point(j), NDArrayIndex.point(0), NDArrayIndex.interval(0, rowIn.size)),
        Nd4j.create(new Array[Double](rowIn.length)))
      inputMask.put(Array[INDArrayIndex](NDArrayIndex.point(j), NDArrayIndex.interval(0, rowIn.size)), Nd4j.ones(rowIn.size))
      predictionMask.put(Array[INDArrayIndex](NDArrayIndex.point(j), NDArrayIndex.interval(0, rowPred.size)), Nd4j.ones(rowPred.size))

      // prediction (output) and decode ARE one-hots though,
      // I couldn't add an embedding layer on top of the decoder and I'm not sure
      // it's a good idea either
      val predOneHot = Array.ofDim[Double](dictSize, rowPred.size)
      val decodeOneHot = Array.ofDim[Double](dictSize, rowPred.size)

      decodeOneHot(2)(0) = 1 // <go> token

      var predIdx = 0
      for (pred <- rowPred) {
        predOneHot(pred.intValue)(predIdx) = 1
        if (predIdx < rowPred.size - 1) { // put the same vals to decode with +1 offset except the last token that is <eos>
          decodeOneHot(pred.intValue)(predIdx + 1) = 1
        }
        predIdx = predIdx + 1
      }

      prediction.put(Array[INDArrayIndex](NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
        NDArrayIndex.interval(0, rowPred.size)), Nd4j.create(predOneHot))
      decode.put(Array[INDArrayIndex](NDArrayIndex.point(j), NDArrayIndex.interval(0, dictSize),
        NDArrayIndex.interval(0, rowPred.size)), Nd4j.create(decodeOneHot))

      i += 1
    }
    currentBatch += 1

    new MultiDataSet(Array[INDArray](input, decode), Array[INDArray](prediction),
      Array[INDArray](inputMask, predictionMask), Array[INDArray](predictionMask))
  }

  override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = {}

  override def resetSupported(): Boolean = false

  override def asyncSupported(): Boolean = true

  override def reset(): Unit = {
    currentBatch = 0
    currentMacroBatch = 0
  }

  def totalBatches(): Int = Math.ceil(corpus.size.asInstanceOf[Double] / batchSize).toInt

  override def next(): MultiDataSet = next(batchSize)

  override def hasNext: Boolean = currentBatch < totalBatches() && (getMacroBatchByCurrentBatch == currentMacroBatch)

  private def getMacroBatchByCurrentBatch = currentBatch / batchesPerMacrobatch

  def batch: Int = currentBatch

  def setCurrentBatch(currentBatch: Int): Unit = {
    this.currentBatch = currentBatch
    currentMacroBatch = getMacroBatchByCurrentBatch
  }

  def hasNextMacrobatch: Boolean = getMacroBatchByCurrentBatch < totalMacroBatches && currentMacroBatch < totalMacroBatches

  def nextMacroBatch(): Unit = {
    currentMacroBatch += 1
  }

  override def getPreProcessor: MultiDataSetPreProcessor = null
}
