package org.dl4scala.datasets.iterator

import org.nd4j.linalg.dataset.DataSet

/**
  * Created by endy on 2017/8/30.
  */
trait DataSetFetcher extends Serializable{
  /**
    * Whether the dataset has more to load
    *
    * @return whether the data applyTransformToDestination has more to load
    */
  def hasMore(): Boolean

  /**
    * Returns the next data applyTransformToDestination
    *
    * @return the next dataset
    */
  def next(): DataSet

  /**
    * Fetches the next dataset. You need to call this
    * to getFromOrigin a new dataset, otherwise {@link #next()}
    * just returns the last data applyTransformToDestination fetch
    *
    * @param numExamples the number of examples to fetch
    */
  def fetch(numExamples: Int): Unit

  /**
    * The number of labels for a dataset
    *
    * @return the number of labels for a dataset
    */
  def totalOutcomes(): Int

  /**
    * The length of a feature vector for an individual example
    *
    * @return the length of a feature vector for an individual example
    */
  def inputColumns(): Int

  /**
    * The total number of examples
    *
    * @return the total number of examples
    */
  def totalExamples(): Int

  /**
    * Returns the fetcher back to the beginning of the dataset
    */
  def reset(): Unit

  /**
    * Direct access to a number represenative of iterating through a dataset
    *
    * @return a cursor similar to an index
    */
  def cursor(): Int
}
