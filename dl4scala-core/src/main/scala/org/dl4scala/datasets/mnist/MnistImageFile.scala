package org.dl4scala.datasets.mnist

import java.io.IOException

/**
  * Created by endy on 16-12-21.
  */
class MnistImageFile(name: String, mode: String) extends MnistDbFile(name, mode){

  private val rows: Int = readInt()
  private val cols: Int = readInt()

  /**
    * Reads the image at the current position.
    *
    * @return matrix representing the image
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readImage: Array[Array[Int]] = {
    val dat: Array[Array[Int]] = Array.ofDim(getRows, getCols)

    0.until(getCols).foreach{i =>
      0.until(getRows).foreach{j =>
        dat(i)(j) = readUnsignedByte
      }
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
    val out: Array[Array[Byte]] = Array.ofDim(nImages, 0)
    var i: Int = 0
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
  def nextImage() = super.next()


  /**
    * Move the cursor to the previous image.
    *
    * @throws IOException
    */
  @throws(classOf[IOException])
  def prevImage() = super.prev()

  override protected def getMagicNumber: Int = 2051

  /**
    * Number of rows per image.
    */
  def getRows: Int = rows

  /**
    * Number of columns per image.
    */
  def getCols: Int = cols

  override def getEntryLength: Int = cols * rows

  override def getHeaderSize: Int = super.getHeaderSize + 8 // to more integers - rows and columns
}
