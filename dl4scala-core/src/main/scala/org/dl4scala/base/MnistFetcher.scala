package org.dl4scala.base

import org.slf4j.Logger
import org.slf4j.LoggerFactory

/**
  * Created by endy on 2017/8/25.
  */
class MnistFetcher {

}

object MnistFetcher {
  protected val log: Logger = LoggerFactory.getLogger(MnistFetcher.getClass)

  protected val LOCAL_DIR_NAME = "MNIST"

  private val trainingFilesURL = "http://benchmark.deeplearn.online/mnist/train-images-idx3-ubyte.gz"
  private val trainingFilesMD5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873"
  private val trainingFilesFilename = "train-images-idx3-ubyte.gz"
  val trainingFilesFilename_unzipped = "train-images-idx3-ubyte"
  private val trainingFileLabelsURL = "http://benchmark.deeplearn.online/mnist/train-labels-idx1-ubyte.gz"
  private val trainingFileLabelsMD5 = "d53e105ee54ea40749a09fcbcd1e9432"
  private val trainingFileLabelsFilename = "train-labels-idx1-ubyte.gz"
  val trainingFileLabelsFilename_unzipped = "train-labels-idx1-ubyte"

  //Test data:
  private val testFilesURL = "http://benchmark.deeplearn.online/mnist/t10k-images-idx3-ubyte.gz"
  private val testFilesMD5 = "9fb629c4189551a2d022fa330f9573f3"
  private val testFilesFilename = "t10k-images-idx3-ubyte.gz"
  val testFilesFilename_unzipped = "t10k-images-idx3-ubyte"
  private val testFileLabelsURL = "http://benchmark.deeplearn.online/mnist/t10k-labels-idx1-ubyte.gz"
  private val testFileLabelsMD5 = "ec29112dd5afa0611ce80d1b7f02629c"
  private val testFileLabelsFilename = "t10k-labels-idx1-ubyte.gz"
  val testFileLabelsFilename_unzipped = "t10k-labels-idx1-ubyte"
}
