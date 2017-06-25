package org.dl4scala.examples.transferlearning.vgg16.dataHelpers

import java.io.{File, IOException}
import java.net.URL

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.BaseImageLoader
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util
import java.util.Random

import org.apache.commons.io.FileUtils
import org.datavec.api.util.ArchiveUtils
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels

/**
  * Created by endy on 2017/6/25.
  */
object FlowerDataSetIterator {
  private val log = org.slf4j.LoggerFactory.getLogger(FlowerDataSetIterator.getClass)

  private val DATA_DIR = new File(System.getProperty("user.home")) + "/dl4jDataDir"
  private val DATA_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
  private val FLOWER_DIR = DATA_DIR + "/flower_photos"

  private val allowedExtensions = BaseImageLoader.ALLOWED_FORMATS
  private val rng = new Random(13)

  private val height = 224
  private val width = 224
  private val channels = 3
  private val numClasses = 5

  private val labelMaker = new ParentPathLabelGenerator
  private var trainData: InputSplit = _
  private var testData: InputSplit = _
  private var batchSize = 0

  @throws(classOf[IOException])
  def trainIterator: DataSetIterator = makeIterator(trainData)

  @throws(classOf[IOException])
  def testIterator: DataSetIterator = makeIterator(testData)

  @throws(classOf[IOException])
  def setup(batchSizeArg: Int, trainPerc: Int): Unit = {
    try
      downloadAndUntar()
    catch {
      case e: IOException =>
        e.printStackTrace()
        log.error("IOException : ", e)
    }

    batchSize = batchSizeArg
    val parentDir = new File(FLOWER_DIR)
    val filesInDir = new FileSplit(parentDir, allowedExtensions, rng)
    val pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker)
    if (trainPerc >= 100)
      throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%." +
        " Test percentage = 100 - training percentage, has to be greater than 0")
    val filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100 - trainPerc)
    trainData = filesInDirSplit(0)
    testData = filesInDirSplit(1)
  }

  @throws(classOf[IOException])
  private def makeIterator(split: InputSplit) = {
    val recordReader = new ImageRecordReader(height, width, channels, labelMaker)
    recordReader.initialize(split)
    val iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses)
    iter.setPreProcessor(TrainedModels.VGG16.getPreProcessor)
    iter
  }

  @throws(classOf[IOException])
  def downloadAndUntar(): Unit = {
    val rootFile = new File(DATA_DIR)
    if (!rootFile.exists) rootFile.mkdir
    val tarFile = new File(DATA_DIR, "flower_photos.tgz")
    if (!tarFile.isFile) {
      log.info("Downloading the flower dataset from " + DATA_URL + "...")
      FileUtils.copyURLToFile(new URL(DATA_URL), tarFile)
    }
    ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath, rootFile.getAbsolutePath)
  }
}
