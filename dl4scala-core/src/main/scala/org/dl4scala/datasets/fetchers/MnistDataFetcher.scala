package org.dl4scala.datasets.fetchers

import java.io.File
import java.util

import org.apache.commons.io.FileUtils
import org.deeplearning4j.base.MnistFetcher
import org.deeplearning4j.datasets.mnist.MnistManager
import org.deeplearning4j.util.MathUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by endy on 16-12-19.
  */
class MnistDataFetcher(binarize: Boolean, train: Boolean, shuffle: Boolean, rngSeed: Long)
      extends BaseDataFetcher {

  protected val TEMP_ROOT: String = System.getProperty("user.home")
  protected val MNIST_ROOT: String = TEMP_ROOT + File.separator + "MNIST" + File.separator

  var man: MnistManager = _
  var order: Array[Int] = _
  var rng: util.Random = _

  init()

  def init(): Unit = {
    if (!mnistExists) new MnistFetcher().downloadAndUntar
    var images: String = null
    var labels: String = null
    if (train) {
      images = MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped
      labels = MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped
      totalExamplesVar = MnistDataFetcher.NUM_EXAMPLES
    }
    else {
      images = MNIST_ROOT + MnistFetcher.testFilesFilename_unzipped
      labels = MNIST_ROOT + MnistFetcher.testFileLabelsFilename_unzipped
      totalExamplesVar = MnistDataFetcher.NUM_EXAMPLES_TEST
    }
    try
      man = new MnistManager(images, labels, train)
    catch {
      case e: Exception =>
        FileUtils.deleteDirectory(new File(MNIST_ROOT))
        new MnistFetcher().downloadAndUntar
        man = new MnistManager(images, labels, train)
    }
    numOutcomesVar = 10
    cursorVar = 0
    inputColumnsVar = man.getImages.getEntryLength
    order = if (train) new Array[Int](MnistDataFetcher.NUM_EXAMPLES) else new Array[Int](MnistDataFetcher.NUM_EXAMPLES_TEST)
    order.indices.foreach(i => order(i) = i)
    rng = new util.Random(rngSeed)
    reset() //Shuffle order
  }


  def this(binarize: Boolean) {
    this(binarize, true, true, System.currentTimeMillis)
  }

  private def mnistExists: Boolean = {
    if (!new File(MNIST_ROOT, MnistFetcher.trainingFilesFilename_unzipped).exists()) false
    else if (!new File(MNIST_ROOT, MnistFetcher.trainingFileLabelsFilename_unzipped).exists())
      false
    else if (!new File(MNIST_ROOT, MnistFetcher.testFilesFilename_unzipped).exists()) false
    else if (!new File(MNIST_ROOT, MnistFetcher.testFileLabelsFilename_unzipped).exists()) false
    else true
  }

  def this() {
    this(true)
  }

  override def fetch(numExamples: Int): Unit = {
    if (!hasMore) throw new IllegalStateException("Unable to getFromOrigin more; there are no more images")

    var featureData: Array[Array[Float]] = Array.ofDim[Float](numExamples, 0)
    var labelData: Array[Array[Float]] = Array.ofDim[Float](numExamples, 0)

    var actualExamples: Int = 0
    var i: Int = 0
    while (i < numExamples && hasMore) {
      val img: Array[Byte] = man.readImageUnsafe(order(cursorVar))
      val label: Int = man.readLabel(order(cursorVar))
      val featureVec: Array[Float] = new Array[Float](img.length)
      featureData(actualExamples) = featureVec
      labelData(actualExamples) = new Array[Float](10)
      labelData(actualExamples)(label) = 1.0f
      var j: Int = 0
      while (j < img.length) {
        val v: Float = img(j).toInt & 0xFF //byte is loaded as signed -> convert to unsigned
        if (binarize) if (v > 30.0f) featureVec(j) = 1.0f
        else featureVec(j) = 0.0f
        else featureVec(j) = v / 255.0f
        j += 1
      }
      actualExamples += 1
      i += 1
      cursorVar += 1
    }

    if (actualExamples < numExamples) {
      featureData = util.Arrays.copyOfRange(featureData, 0, actualExamples)
      labelData = util.Arrays.copyOfRange(labelData, 0, actualExamples)
    }

    val features: INDArray = Nd4j.create(featureData)
    val labels: INDArray = Nd4j.create(labelData)
    currVar = new DataSet(features, labels)
  }

  override def reset() {
    cursorVar = 0
    if (shuffle) MathUtils.shuffleArray(order, rng)
  }

  override def next: DataSet = {
    val next: DataSet = super.next
    next
  }
}

object MnistDataFetcher {
  val NUM_EXAMPLES: Int = 60000
  val NUM_EXAMPLES_TEST: Int = 10000
}

