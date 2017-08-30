package org.dl4scala.datasets.iterator

import java.util

import org.dl4scala.datasets.fetchers.BaseDataFetcher
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

/**
  * Created by endy on 2017/8/30.
  */
class BaseDatasetIterator(var batchValue: Int, var numExamplesValue: Int, fetcher: BaseDataFetcher) extends DataSetIterator{

  protected var preProcessor: DataSetPreProcessor = _

  if (numExamplesValue < 0) numExamplesValue = fetcher.totalExamples()

  override def cursor(): Int = fetcher.cursor()

  override def next(num: Int): DataSet = {
    fetcher.fetch(num)
    val next = fetcher.next()
    if (preProcessor != null) preProcessor.preProcess(next)
    next
  }

  override def setPreProcessor(dataSetPreProcessor: DataSetPreProcessor): Unit = {
    preProcessor = dataSetPreProcessor
  }

  override def getPreProcessor: DataSetPreProcessor = preProcessor

  override def totalOutcomes(): Int = fetcher.totalOutcomes()

  override def getLabels: util.List[String] = null

  override def inputColumns(): Int = fetcher.inputColumns()

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = batchValue

  override def reset(): Unit = fetcher.reset()

  override def totalExamples(): Int = fetcher.totalExamples()

  override def numExamples(): Int = numExamplesValue

  override def next(): DataSet = {
    fetcher.fetch(batchValue)
    val result = fetcher.next()
    if (preProcessor != null) preProcessor.preProcess(result)
    result
  }

  override def hasNext: Boolean = fetcher.hasMore() && fetcher.cursor() < numExamplesValue
}
