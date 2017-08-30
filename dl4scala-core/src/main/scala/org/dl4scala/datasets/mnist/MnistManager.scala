package org.dl4scala.datasets.mnist

import java.io.{BufferedWriter, FileWriter, IOException}

import org.dl4scala.datasets.fetchers.MnistDataFetcher

/**
  * Created by endy on 2017/8/28.
  */
class MnistManager(imagesFile: String, labelsFile: String, numExamples: Int)  {

  var images: MnistImageFile = _
  private var labels: MnistLabelFile = _

  private var imagesArr: Array[Array[Byte]] = _
  private var labelsArr: Array[Int] = _

  def this(imagesFile: String, labelsFile: String, train: Boolean) =
    this(imagesFile, labelsFile, if (train) MnistDataFetcher.NUM_EXAMPLES else MnistDataFetcher.NUM_EXAMPLES_TEST)

  def this(imagesFile: String, labelsFile: String) = this(imagesFile, labelsFile, true)


  if (imagesFile.isEmpty) {
    images = new MnistImageFile(imagesFile, "r")
    imagesArr = images.readImagesUnsafe(numExamples)
  }
  if (labelsFile.isEmpty) {
    labels = new MnistLabelFile(labelsFile, "r")
    labelsArr = labels.readLabels(numExamples)
  }

  /**
    * Reads the current image.
    *
    * @return matrix
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readImage: Array[Array[Int]] = {
    if (images == null) throw new IllegalStateException("Images file not initialized.")
    images.readImage
  }

  def readImageUnsafe(i: Int): Array[Byte] = imagesArr(i)

  /**
    * Set the position to be read.
    *
    * @param index
    */
  def setCurrent(index: Int): Unit = {
    images.setCurrentIndex(index)
    labels.setCurrentIndex(index)
  }

  /**
    * Reads the current label.
    *
    * @return int
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readLabel: Int = {
    if (labels == null) throw new IllegalStateException("labels file not initialized.")
    labels.readLabel
  }

  def readLabel(i: Int): Int = labelsArr(i)

  /**
    * Get the underlying images file as {@link MnistImageFile}.
    *
    * @return { @link MnistImageFile}.
    */
  def getImages: MnistImageFile = images

  /**
    * Get the underlying labels file as {@link MnistLabelFile}.
    *
    * @return { @link MnistLabelFile}.
    */
  def getLabels: MnistLabelFile = labels

  /**
    * Close any resources opened by the manager.
    */
  def close(): Unit = {
    if (images != null) {
      try
        images.close()
      catch {
        case e: IOException =>

      }
      images = null
    }
    if (labels != null) {
      try
        labels.close()
      catch {
        case e: IOException =>
      }
      labels = null
    }
  }
}

object MnistManager {
  private val HEADER_SIZE = 8

  @throws(classOf[IOException])
  def writeImageToPpm(image: Array[Array[Int]], ppmFileName: String): Unit = {
    try {
      val ppmOut = new BufferedWriter(new FileWriter(ppmFileName))
      val rows = image.length
      val cols = image(0).length
      ppmOut.write("P3\n")
      ppmOut.write("" + rows + " " + cols + " 255\n")

      (0 until rows).foreach{ i =>
        val s = new StringBuilder
        (0 until cols).foreach(j => s.append(image(i)(j) + " " + image(i)(j) + " " + image(i)(j) + "  "))
        ppmOut.write(s.toString)
      }
      ppmOut.close()
    } catch {
      case e: Exception => println("BufferedWriter error" + e.printStackTrace())
    }
  }
}