package org.dl4scala.datasets.iterator.impl

import org.dl4scala.datasets.fetchers.MnistDataFetcher
import org.dl4scala.datasets.iterator.BaseDatasetIterator
import java.io.IOException

/**
  * Mnist data applyTransformToDestination iterator.
  *
  */
@throws(classOf[IOException])
class MnistDataSetIterator(override val batch: Int,
                           override val numExamples: Int,
                           val binarize: Boolean,
                           val train: Boolean,
                           val shuffle: Boolean,
                           val rngSeed: Long) extends
  BaseDatasetIterator(batch, numExamples, new MnistDataFetcher(binarize, train, shuffle, rngSeed)){

  /** Get the specified number of examples for the MNIST training data set.
    *
    * @param batch       the batch size of the examples
    * @param numExamples the overall number of examples
    * @param binarize    whether to binarize mnist or not
    * @throws IOException
    */
  @throws(classOf[IOException])
  def this(batch: Int, numExamples: Int, binarize: Boolean) {
    this(batch, numExamples, binarize, true, false, 0)
  }

  @throws(classOf[IOException])
  def this(batch: Int, numExamples: Int) {
    this(batch, numExamples, false)
  }

  /** Constructor to get the full MNIST data set (either test or train sets) without binarization
    *  (i.e., just normalization
    * into range of 0 to 1), with shuffling based on a random seed.
    *
    * @param batchSize
    * @param train
    * @throws IOException
    */
  @throws(classOf[IOException])
  def this(batchSize: Int, train: Boolean, seed: Int) {
    this(batchSize, if (train) MnistDataFetcher.NUM_EXAMPLES else MnistDataFetcher.NUM_EXAMPLES_TEST,
      false, train, true, seed)
  }
}