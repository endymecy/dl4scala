package org.dl4scala.datasets.mnist

import java.io.IOException

/**
  * Created by endy on 16-12-21.
  */
class MnistLabelFile(name: String, mode: String) extends MnistDbFile(name, mode){
  /**
    * Reads the integer at the current position.
    *
    * @return integer representing the label
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readLabel: Int = readUnsignedByte

  /** Read the specified number of labels from the current position */
  @throws(classOf[IOException])
  def readLabels(num: Int): Array[Int] = {
    val out: Array[Int] = new Array[Int](num)

    0.until(num).foreach(i => out(i) = readLabel)

    out
  }

  protected def getMagicNumber: Int = 2049
}
