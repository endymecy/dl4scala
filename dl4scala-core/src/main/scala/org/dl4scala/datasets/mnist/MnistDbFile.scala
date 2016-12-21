package org.dl4scala.datasets.mnist

import java.io.{IOException, RandomAccessFile}

/**
  * Created by endy on 16-12-21.
  */
abstract class MnistDbFile(name: String, mode: String) extends RandomAccessFile(name, mode){

  private val count: Int = if (getMagicNumber != readInt)
    throw new RuntimeException("This MNIST DB file " + name + " should start with the number " +
      getMagicNumber + ".") else readInt

  protected def getMagicNumber: Int

  /**
    * The current entry index.
    *
    * @return long
    */
  @throws(classOf[IOException])
  def getCurrentIndex: Long = (getFilePointer - getHeaderSize) / getEntryLength + 1

  /**
    * Set the required current entry index.
    *
    * @param curr
    * the entry index
    */
  def setCurrentIndex(curr: Long) {
    try
      if (curr < 0 || curr > count) throw new RuntimeException(curr + " is not in the range 0 to " + count)
      seek(getHeaderSize + curr * getEntryLength)

    catch {
      case e: IOException => {
        throw new RuntimeException(e)
      }
    }
  }

  def getHeaderSize: Int = 8 // two integers

  /**
    * Number of bytes for each entry.
    * Defaults to 1.
    *
    * @return int
    */
  def getEntryLength: Int = 1

  /**
    * Move to the next entry.
    *
    */
  @throws(classOf[IOException])
  def next() {
    if (getCurrentIndex < count) skipBytes(getEntryLength)
  }

  /**
    * Move to the previous entry.
    */
  @throws(classOf[IOException])
  def prev() {
    if (getCurrentIndex > 0) seek(getFilePointer - getEntryLength)
  }

  def getCount: Int = count
}
