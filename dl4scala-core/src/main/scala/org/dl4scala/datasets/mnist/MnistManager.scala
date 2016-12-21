package org.dl4scala.datasets.mnist

import java.io.{BufferedWriter, FileWriter, IOException}

import org.dl4scala.datasets.fetchers.MnistDataFetcher

/**
  * Created by endy on 16-12-20.
  */
class MnistManager(imagesFile: String, labelsFile: String, train: Boolean) {

  private var images: MnistImageFile = if (imagesFile != null)
    new MnistImageFile(imagesFile, "r") else null

  private var labels: MnistLabelFile = if(labelsFile != null)
    new MnistLabelFile(labelsFile, "r") else null

  private val imagesArr: Array[Array[Byte]] = if (imagesFile != null && train)
    images.readImagesUnsafe(MnistDataFetcher.NUM_EXAMPLES) else
    images.readImagesUnsafe(MnistDataFetcher.NUM_EXAMPLES_TEST)

  private val labelsArr: Array[Int] = if (labelsFile != null && train)
    labels.readLabels(MnistDataFetcher.NUM_EXAMPLES) else
    labels.readLabels(MnistDataFetcher.NUM_EXAMPLES_TEST)


  def this(imagesFile: String, labelsFile: String) {
    this(imagesFile, labelsFile, true)
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
    else images.readImage
  }

  def readImageUnsafe(i: Int): Array[Byte] = imagesArr(i)

  /**
    * Set the position to be read.
    *
    * @param index index
    */
  def setCurrent(index: Int) {
    images.setCurrentIndex(index)
    labels.setCurrentIndex(index)
  }

  /**
    * Reads the current label.
    *
    * @return int
    * @throws IOException throw Exception
    */
  @throws(classOf[IOException])
  def readLabel: Int = {
    if (labels == null) throw new IllegalStateException("labels file not initialized.")
    labels.readLabel
  }

  def readLabel(i: Int): Int = labelsArr(i)

  /**
    * Get the underlying images file.
    *
    * @return { @link MnistImageFile}.
    */
  def getImages: MnistImageFile = images

  /**
    * Get the underlying labels file.
    *
    * @return { @link MnistLabelFile}.
    */
  def getLabels: MnistLabelFile = labels

  /**
    * Close any resources opened by the manager.
    */
  def close() {
    if (images != null) {
      try
        images.close()
      catch {
        case e: IOException => //
      }
      images = null
    }
    if (labels != null) {
      try
        labels.close()
      catch {
        case e: IOException => //
      }
      labels = null
    }
  }
}

object MnistManager {
  private val HEADER_SIZE: Int = 8

  /**
    * Writes the given image in the given file using the PPM data format.
    *
    * @param image
    * @param ppmFileName
    * @throws IOException
    */
  @throws(classOf[IOException])
  def writeImageToPpm(image: Array[Array[Int]], ppmFileName: String) {
      val ppmOut: BufferedWriter = new BufferedWriter(new FileWriter(ppmFileName))
      val rows: Int = image.length
      val cols: Int = image(0).length

      ppmOut.write("P3\n")
      ppmOut.write("" + rows + " " + cols + " 255\n")

      0.until(rows).foreach{i =>
        val s: StringBuilder = new StringBuilder
        0.until(cols).foreach{j =>
          s.append(image(i)(j) + " " + image(i)(j) + " " + image(i)(j) + "  ")
        }
        ppmOut.write(s.toString)
      }
      ppmOut.close()
  }
}
