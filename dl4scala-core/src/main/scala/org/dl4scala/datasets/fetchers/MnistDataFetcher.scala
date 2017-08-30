package org.dl4scala.datasets.fetchers

import java.io.{File, IOException}
import java.util

import org.apache.commons.io.FileUtils
import org.dl4scala.base.MnistFetcher
import org.dl4scala.datasets.mnist.MnistManager
import org.dl4scala.util.MathUtils
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.util.Random


/**
  * Created by endy on 2017/8/25.
  */
class MnistDataFetcher(binarize: Boolean, train: Boolean, shuffle: Boolean, rngSeed: Long) extends BaseDataFetcher {
  import MnistDataFetcher._
  protected var man: MnistManager = _
  protected var order: Array[Int] = _
  protected var rng: Random = _
  protected var oneIndexed = false
  protected var fOrder = false //MNIST is C order, EMNIST is F order

  if (!mnistExists) new MnistFetcher().downloadAndUntar()
  var images: String = _
  var labels: String = _

  if (train) {
    images = MNIST_ROOT + MnistFetcher.trainingFilesFilename_unzipped
    labels = MNIST_ROOT + MnistFetcher.trainingFileLabelsFilename_unzipped
    totalExamplesValue = NUM_EXAMPLES
  }
  else {
    images = MNIST_ROOT + MnistFetcher.testFilesFilename_unzipped
    labels = MNIST_ROOT + MnistFetcher.testFileLabelsFilename_unzipped
    totalExamplesValue = NUM_EXAMPLES_TEST
  }

  try
    man = new MnistManager(images, labels, train)
  catch {
    case e: Exception =>
      FileUtils.deleteDirectory(new File(MNIST_ROOT))
      new MnistFetcher().downloadAndUntar()
      man = new MnistManager(images, labels, train)
  }

  numOutcomes = 10
  cursorValue = 0
  inputColumnsValue = man.getImages.getEntryLength

  if (train) order = new Array[Int](NUM_EXAMPLES)
  else order = new Array[Int](NUM_EXAMPLES_TEST)

  var i: Int = 0
  while (i < order.length) {
    order(i) = i
    i += 1
  }
  rng = new Random(rngSeed)


  @throws(classOf[IOException])
  def this(binarize: Boolean) = this(binarize, true, true, System.currentTimeMillis())

  @throws(classOf[IOException])
  def this() = this(true)

  override def fetch(numExamples: Int): Unit = {
    if (!hasMore) throw new IllegalStateException("Unable to getFromOrigin more; there are no more images")

    var featureData = Array.ofDim[Float](numExamples, 0)
    var labelData = Array.ofDim[Float](numExamples, 0)

    var actualExamples = 0
    var working: Array[Byte] = null

    var i = 0
    var break = false
    while (i < numExamples && !break) {
      if (!hasMore) break = true

      var img = man.readImageUnsafe(order(cursorValue))
      if (fOrder) {
        // EMNIST requires F order to C order//EMNIST requires F order to C order
        if (working == null) working = new Array[Byte](28 * 28)
        (0 until 28 * 28).foreach{j =>
          working(j) = img(28 * (j % 28) + j / 28)
        }

        val temp: Array[Byte] = img
        img = working
        working = temp
      }

      var label = man.readLabel(order(cursorValue))
      if (oneIndexed) { //For some inexplicable reason, Emnist LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
        // is indexed (0 to nClasses-1) :/
        label -= 1
      }
      val featureVec = new Array[Float](img.length)
      featureData(actualExamples) = featureVec
      labelData(actualExamples) = new Array[Float](numOutcomes)
      labelData(actualExamples)(label) = 1.0f

      img.indices.foreach{j =>
        val v = img(j).asInstanceOf[Int] & 0xFF //byte is loaded as signed -> convert to unsigned
        if (binarize) if (v > 30.0f) featureVec(j) = 1.0f
        else featureVec(j) = 0.0f
        else featureVec(j) = v / 255.0f
      }

      actualExamples += 1

      i = i + 1
      cursorValue = cursorValue + 1
    }

    if (actualExamples < numExamples) {
      featureData = util.Arrays.copyOfRange(featureData, 0, actualExamples)
      labelData = util.Arrays.copyOfRange(labelData, 0, actualExamples)
    }

    val features = Nd4j.create(featureData)
    val labels = Nd4j.create(labelData)
    curr = new DataSet(features, labels)
  }

  private def mnistExists(): Boolean = { //Check 4 files:
    var f = new File(MNIST_ROOT, MnistFetcher.trainingFilesFilename_unzipped)
    if (!f.exists) false
    f = new File(MNIST_ROOT, MnistFetcher.trainingFileLabelsFilename_unzipped)
    if (!f.exists) false
    f = new File(MNIST_ROOT, MnistFetcher.testFilesFilename_unzipped)
    if (!f.exists) false
    f = new File(MNIST_ROOT, MnistFetcher.testFileLabelsFilename_unzipped)
    if (!f.exists) false
    true
  }

  override def reset(): Unit = {
    cursorValue = 0
    curr = null
    if (shuffle) MathUtils.shuffleArray(order, rng)
  }

  override def next(): DataSet = {
    val next = super.next()
    next
  }
}

object MnistDataFetcher {
  val NUM_EXAMPLES = 60000
  val NUM_EXAMPLES_TEST = 10000
  protected val TEMP_ROOT: String = System.getProperty("user.home")
  protected val MNIST_ROOT: String = TEMP_ROOT + File.separator + "MNIST" + File.separator

}
