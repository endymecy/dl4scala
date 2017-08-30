package org.dl4scala.datasets.iterator.impl

import java.io.IOException

import org.dl4scala.datasets.fetchers.MnistDataFetcher
import org.dl4scala.datasets.iterator.BaseDatasetIterator

/**
  * Created by endy on 2017/8/30.
  */
/**
  * Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
  * @param batch Size of each patch
  * @param numExamples total number of examples to load
  * @param binarize whether to binarize the data or not (if false: normalize in range 0 to 1)
  * @param train Train vs. test set
  * @param shuffle whether to shuffle the examples
  * @param rngSeed random number generator seed to use when shuffling examples
  */
@throws(classOf[IOException])
class MnistDataSetIterator(batch: Int, numExamples: Int, binarize: Boolean,
                           train: Boolean, shuffle: Boolean, rngSeed: Long)
  extends BaseDatasetIterator(batch, numExamples, new MnistDataFetcher(binarize, train, shuffle, rngSeed)){

  /**
    * Get the specified number of examples for the MNIST training data set.
    * @param batch the batch size of the examples
    * @param numExamples the overall number of examples
    * @param binarize whether to binarize mnist or not
    * @throws IOException
    */
  @throws(classOf[IOException])
  def this(batch: Int, numExamples: Int, binarize: Boolean) = this(batch, numExamples, binarize, true, false, 0)


  def this(batch: Int, numExamples: Int) = this(batch, numExamples, false)


  /**
    * Constructor to get the full MNIST data set (either test or train sets)
    * without binarization (i.e., just normalization
    * into range of 0 to 1), with shuffling based on a random seed.
    * @param batchSize
    * @param train
    * @throws IOException
    */
  @throws(classOf[IOException])
  def this(batchSize: Int, train: Boolean, seed: Int) =
    this(batchSize, if (train) MnistDataFetcher.NUM_EXAMPLES else MnistDataFetcher.NUM_EXAMPLES_TEST, false, train,
    true, seed)
}
