package org.dl4scala.datasets.fetchers

import java.io.File
import java.util

import org.apache.commons.io.FileUtils
import org.dl4scala.base.MnistFetcher
import org.dl4scala.datasets.mnist.MnistManager
import org.deeplearning4j.util.MathUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by endy on 16-12-19.
  */
class MnistDataFetcher(val binarize: Boolean,
                       val train: Boolean,
                       val shuffle: Boolean,
                       val rngSeed: Long) extends BaseDataFetcher {

  private val TEMP_ROOT: String = System.getProperty("user.home")
  private val MNIST_ROOT: String = TEMP_ROOT + File.separator + "MNIST" + File.separator
  private var man: MnistManager = _
  private var order: Array[Int] = _
  private var rng: util.Random = _

  init()

  def init(): Unit = {
    if (!mnistExists) MnistFetcher.downloadAndUntar

    val (images, labels) = if (train) {
      totalExamples = MnistDataFetcher.NUM_EXAMPLES
      (MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped,
        MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped)
    } else {
      totalExamples = MnistDataFetcher.NUM_EXAMPLES_TEST
      (MNIST_ROOT + MnistFetcher.testFilesFilename_unzipped,
        MNIST_ROOT + MnistFetcher.testFileLabelsFilename_unzipped)
    }

    try
      man = new MnistManager(images, labels, train)
    catch {
      case e: Exception =>
        FileUtils.deleteDirectory(new File(MNIST_ROOT))
        MnistFetcher.downloadAndUntar
        man = new MnistManager(images, labels, train)
    }
    numOutcomes = 10
    cursor = 0
    inputColumns = man.getImages.getEntryLength

    order = if (train)
      new Array[Int](MnistDataFetcher.NUM_EXAMPLES)
    else new Array[Int](MnistDataFetcher.NUM_EXAMPLES_TEST)

    order.indices.foreach(i => order(i) = i)
    rng = new util.Random(rngSeed)
    reset() //Shuffle order
  }

  def this(binarize: Boolean) {
    this(binarize, true, true, System.currentTimeMillis)
  }

  def this() = this(true)

  private def mnistExists: Boolean = {
    if (!new File(MNIST_ROOT, MnistFetcher.trainingFilesFilename_unzipped).exists())
      false
    else if (!new File(MNIST_ROOT, MnistFetcher.trainingFileLabelsFilename_unzipped).exists())
      false
    else if (!new File(MNIST_ROOT, MnistFetcher.testFilesFilename_unzipped).exists())
      false
    else if (!new File(MNIST_ROOT, MnistFetcher.testFileLabelsFilename_unzipped).exists())
      false
    else true
  }

  override def fetch(numExamples: Int): Unit = {
    if (!hasMore) throw new IllegalStateException("Unable to getFromOrigin more; there are no more images")

    var featureData: Array[Array[Float]] = Array.ofDim[Float](numExamples, 0)
    var labelData: Array[Array[Float]] = Array.ofDim[Float](numExamples, 0)

    var actualExamples: Int = 0
    var i: Int = 0
    while (i < numExamples && hasMore) {
      val img: Array[Byte] = man.readImageUnsafe(order(cursor))
      val label: Int = man.readLabel(order(cursor))
      val featureVec: Array[Float] = new Array[Float](img.length)
      featureData(actualExamples) = featureVec
      labelData(actualExamples) = new Array[Float](10)
      labelData(actualExamples)(label) = 1.0f

      img.indices.foreach{j =>
        val v: Float = img(j).toInt & 0xFF //byte is loaded as signed -> convert to unsigned
        if (binarize) if (v > 30.0f) featureVec(j) = 1.0f
        else featureVec(j) = 0.0f
        else featureVec(j) = v / 255.0f
      }
      actualExamples += 1
      i += 1
      cursor += 1
    }

    if (actualExamples < numExamples) {
      featureData = util.Arrays.copyOfRange(featureData, 0, actualExamples)
      labelData = util.Arrays.copyOfRange(labelData, 0, actualExamples)
    }

    val features: INDArray = Nd4j.create(featureData)
    val labels: INDArray = Nd4j.create(labelData)
    curr = new DataSet(features, labels)
  }

  override def reset() {
    cursor = 0
    if (shuffle) MathUtils.shuffleArray(order, rng)
  }

  override def next: DataSet = super.next

}

object MnistDataFetcher {
  val NUM_EXAMPLES: Int = 60000
  val NUM_EXAMPLES_TEST: Int = 10000

}

