package org.dl4scala.datasets.mnist

import java.io.IOException

/**
  * Created by endy on 2017/8/28.
  */
class MnistImageFile(name: String, mode: String) extends MnistDbFile(name, mode) {
  /**
    * MNIST DB files start with unique integer number.
    *
    * @return integer number that should be found in the beginning of the file.
    */
  override def getMagicNumber: Int = 2051

  private val rows = readInt()
  private val cols = readInt()

  /**
    * Reads the image at the current position.
    *
    * @return matrix representing the image
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readImage: Array[Array[Int]] = {
    val dat = Array.ofDim[Int](getRows, getCols)
    var i = 0
    while (i < getCols) {
      var j = 0
      while (j < getRows) {
        dat(i)(j) = readUnsignedByte
        j += 1
      }
      i += 1
    }
    dat
  }

  /** Read the specified number of images from the current position, to a byte[nImages][rows*cols]
    * Note that MNIST data set is stored as unsigned bytes; this method returns signed bytes without conversion
    * (i.e., same bits, but requires conversion before use)
    *
    * @param nImages Number of images
    */
  @throws(classOf[IOException])
  def readImagesUnsafe(nImages: Int): Array[Array[Byte]] = {
    val out = Array.ofDim[Byte](nImages, 0)
    var i = 0
    while (i < nImages) {
      out(i) = new Array[Byte](rows * cols)
      read(out(i))
      i += 1
    }
    out
  }


  /**
    * Move the cursor to the next image.
    *
    * @throws IOException
    */
  @throws(classOf[IOException])
  def nextImage(): Unit = {
    super.next()
  }

  /**
    * Move the cursor to the previous image.
    *
    * @throws IOException
    */
  @throws(classOf[IOException])
  def prevImage(): Unit = {
    super.prev()
  }

  /**
    * Number of rows per image.
    *
    * @return int
    */
  def getRows: Int = rows

  /**
    * Number of columns per image.
    *
    * @return int
    */
  def getCols: Int = cols


  override def getEntryLength: Int = cols * rows

  override def getHeaderSize: Int = super.getHeaderSize + 8 // to more integers - rows and columns
}
