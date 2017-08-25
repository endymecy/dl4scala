package org.dl4scala.datasets

import org.dl4scala.datasets.fetchers.MnistDataFetcher
import org.nd4j.linalg.dataset.DataSet

/**
  * Created by endy on 2017/8/25.
  */
object DataSets {

  def mnist(): DataSet = {
    mnist(60000)
  }

  def mnist(num : Int): DataSet = {
    try {
      val fetcher: MnistDataFetcher = new MnistDataFetcher()
      fetcher.fetch(num)
      fetcher.next()
    } catch {
      case e: Exception =>
        throw new RuntimeException(e)
    }
  }

  def iris(): DataSet = {
    mnist(60000)
  }

  def iris(num : Int): DataSet = {
    try {
      val fetcher: IrisDataFetcher = new IrisDataFetcher()
      fetcher.fetch(num)
      fetcher.next()
    } catch {
      case e: Exception =>
        throw new RuntimeException(e)
    }
  }
}
