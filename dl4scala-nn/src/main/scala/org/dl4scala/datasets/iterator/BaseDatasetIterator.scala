package org.dl4scala.datasets.iterator

import org.dl4scala.datasets.fetchers.BaseDataFetcher
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util

/**
  * Baseline implementation includes
  * control over the data fetcher and some basic
  * getters for metadata
  *
  */
class BaseDatasetIterator(val batch: Int, val numExamples: Int, val fetcher: BaseDataFetcher)
  extends DataSetIterator {

  private val num: Int = if (numExamples < 0) fetcher.totalExamples else numExamples

  protected var preProcessor: DataSetPreProcessor = _

  override def hasNext: Boolean ={
    fetcher.hasMore && fetcher.cursor < num
  }

  override def next: DataSet = {
    fetcher.fetch(batch)
    val result: DataSet = fetcher.next
    if (preProcessor != null) preProcessor.preProcess(result)
    result
  }

  override def next(num: Int): DataSet = {
    fetcher.fetch(num)
    val next: DataSet = fetcher.next
    if (preProcessor != null) preProcessor.preProcess(next)
    next
  }

  override def remove() {
    throw new UnsupportedOperationException
  }

  override def totalExamples: Int = fetcher.totalExamples

  override def inputColumns: Int = fetcher.inputColumns

  override def totalOutcomes: Int = fetcher.totalOutcomes

  override def resetSupported: Boolean = true

  override def asyncSupported: Boolean = true

  override def reset() {
    fetcher.reset()
  }

  override def cursor: Int = fetcher.cursor

  override def setPreProcessor(preProcessor: DataSetPreProcessor) {
    this.preProcessor = preProcessor
  }

  override def getLabels: util.List[String] = null

  override def getPreProcessor: DataSetPreProcessor = preProcessor
}