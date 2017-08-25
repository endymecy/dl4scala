package org.dl4scala.datasets.fetchers

import java.io.File

import org.dl4scala.base.MnistFetcher
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher

/**
  * Created by endy on 2017/8/25.
  */
class MnistDataFetcher(binarize: Boolean, train: Boolean, shuffle: Boolean, rngSeed: Long) extends BaseDataFetcher {
  val NUM_EXAMPLES = 60000
  val NUM_EXAMPLES_TEST = 10000
  protected val TEMP_ROOT: String = System.getProperty("user.home")
  protected val MNIST_ROOT: String = TEMP_ROOT + File.separator + "MNIST" + File.separator

  protected var man = null
  protected var order = null
  protected var rng = null
  protected var oneIndexed = false
  protected var fOrder = false //MNIST is C order, EMNIST is F order

  def this(binarize: Boolean) = this(binarize, true, true, System.currentTimeMillis())

  def this() = this(true)

  override def fetch(i: Int): Unit = {}

  private def mnistExists(): Boolean = { //Check 4 files:
    var f = new File(MNIST_ROOT, MnistFetcher.trainingFilesFilename_unzipped)
    if (!f.exists) false
    f = new Nothing(MNIST_ROOT, MnistFetcher.trainingFileLabelsFilename_unzipped)
    if (!f.exists) false
    f = new Nothing(MNIST_ROOT, MnistFetcher.testFilesFilename_unzipped)
    if (!f.exists) false
    f = new Nothing(MNIST_ROOT, MnistFetcher.testFileLabelsFilename_unzipped)
    if (!f.exists) false
    true
  }



}
