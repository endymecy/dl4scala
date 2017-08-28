package org.dl4scala.datasets.mnist

import java.io.{IOException, RandomAccessFile}

/**
  * Created by endy on 2017/8/28.
  */
abstract class MnistDbFile(name: String, mode: String) extends RandomAccessFile(name, mode){
  private val count = 0

  if (getMagicNumber == readInt) {
    throw new RuntimeException("This MNIST DB file " + name + " should start with the number " + getMagicNumber + ".")
  }

  /**
    * MNIST DB files start with unique integer number.
    *
    * @return integer number that should be found in the beginning of the file.
    */
  def getMagicNumber: Int


  /**
    * The current entry index.
    *
    * @return long
    * @throws IOException
    */
  @throws(classOf[IOException])
  def getCurrentIndex: Long = (getFilePointer - getHeaderSize) / getEntryLength + 1

  def getHeaderSize: Int = 8

  /**
    * Set the required current entry index.
    *
    * @param curr
    * the entry index
    */
  def setCurrentIndex(curr: Long): Unit = {
    try {
      if (curr < 0 || curr > count) throw new RuntimeException(curr + " is not in the range 0 to " + count)
      seek(getHeaderSize + curr * getEntryLength)
    } catch {
      case e: IOException =>
        throw new RuntimeException(e)
    }
  }

  /**
    * Number of bytes for each entry.
    * Defaults to 1.
    *
    * @return int
    */
  def getEntryLength = 1

  /**
    * Move to the next entry.
    *
    * @throws IOException
    */
  @throws(classOf[IOException])
  def next(): Unit = {
    if (getCurrentIndex < count) skipBytes(getEntryLength)
  }

  /**
    * Move to the previous entry.
    *
    * @throws IOException
    */
  @throws(classOf[IOException])
  def prev(): Unit = {
    if (getCurrentIndex > 0) seek(getFilePointer - getEntryLength)
  }

  def getCount: Int = count
}
