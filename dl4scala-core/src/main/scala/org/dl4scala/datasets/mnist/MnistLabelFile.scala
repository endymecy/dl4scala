package org.dl4scala.datasets.mnist

import java.io.IOException

/**
  * MNIST database label file.
  *
  * Created by endy on 2017/8/28.
  */
class MnistLabelFile(name: String, mode: String) extends MnistDbFile(name, mode){
  /**
    * MNIST DB files start with unique integer number.
    *
    * @return integer number that should be found in the beginning of the file.
    */
  override def getMagicNumber: Int = 2049

  /**
    * Reads the integer at the current position.
    *
    * @return integer representing the label
    * @throws IOException
    */
  @throws(classOf[IOException])
  def readLabel: Int = readUnsignedByte

  /** Read the specified number of labels from the current position */
  @throws[IOException]
  def readLabels(num: Int): Array[Int] = {
    val out = new Array[Int](num)
    var i = 0
    while (i < num) {
      out(i) = readLabel
      i += 1
    }
    out
  }
}
