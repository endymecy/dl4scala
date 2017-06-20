package org.dl4scala.examples.recurrent.character

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable
import scala.util.Random
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/20.
  */
class CharacterIterator(textFilePath: String, textFileEncoding: Charset , miniBatchSize: Int, exampleLength: Int,
                        validCharacters: Array[Char], rng: Random) extends DataSetIterator{

  //Maps each character to an index ind the input/output
  private val charToIdxMap = new mutable.OpenHashMap[Character, Integer]()
  private var fileCharacters: Array[Char] = _

  //Offsets for the start of each example
  private var exampleStartOffsets = new mutable.Queue[Integer]()

  init()

  def init(): Unit = {
    if (!new File(textFilePath).exists) throw new IOException("Could not access file (does not exist): " + textFilePath)
    if (miniBatchSize <= 0) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)")

    validCharacters.indices.foreach(i => charToIdxMap.put(validCharacters(i), i))

    // Load file and convert contents to a char[]
    val newLineValid: Boolean = charToIdxMap.contains('\n')
    val lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    var maxSize: Int = lines.size  // add lines.size() to account for newline characters at end of each line

    for (s: String <- lines.asScala) {
      maxSize += s.length
    }

    val characters = new Array[Char](maxSize)
    var currIdx = 0

    for (s: String <- lines.asScala) {
      val thisLine = s.toCharArray
      for (aThisLine: Char <- thisLine) {
        if (charToIdxMap.contains(aThisLine)) {
          characters(currIdx) = aThisLine
          currIdx += 1
        }
      }
      if(newLineValid) characters(currIdx) = '\n'
      currIdx += 1
    }

    if (currIdx == characters.length) fileCharacters = characters
    else fileCharacters = util.Arrays.copyOfRange(characters, 0, currIdx)

    if( exampleLength >= fileCharacters.length ) throw new IllegalArgumentException("exampleLength="+exampleLength
      +" cannot exceed number of valid characters in file ("+fileCharacters.length+")")

    val nRemoved: Int = maxSize - fileCharacters.length
    System.out.println("Loaded and converted file: " + fileCharacters.length +
      " valid characters of " + maxSize + " total characters (" + nRemoved + " removed)")

    initializeOffsets()
  }

  def convertIndexToCharacter(idx: Int): Char = validCharacters(idx)

  def convertCharacterToIndex(c: Char): Int = charToIdxMap(c)

  def getRandomCharacter: Char = validCharacters((rng.nextDouble * validCharacters.length).asInstanceOf[Int])

  override def cursor(): Int = totalExamples() - exampleStartOffsets.size

  override def next(num: Int): DataSet = {
    if (exampleStartOffsets.isEmpty) throw new NoSuchElementException
    val currMinibatchSize = Math.min(num, exampleStartOffsets.size)

    // Allocate space:
    // Note the order here:
    // dimension 0 = number of examples in minibatch
    // dimension 1 = size of each vector (i.e., number of characters)
    // dimension 2 = length of each time series/example
    val input = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length, exampleLength), 'f')
    val labels = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length, exampleLength), 'f')

    (0 until currMinibatchSize).foreach{i =>
      val startIdx: Integer = exampleStartOffsets.dequeueFirst(_ => true).get
      val endIdx = startIdx + exampleLength
      var currCharIdx = charToIdxMap(fileCharacters(startIdx)) //Current input
      var c: Int = 0
      (startIdx+1 until endIdx).foreach{j =>
        val nextCharIdx = charToIdxMap(fileCharacters(j)) //Next character to predict

        input.putScalar(Array[Int](i, currCharIdx, c), 1.0)
        labels.putScalar(Array[Int](i, nextCharIdx, c), 1.0)
        currCharIdx = nextCharIdx
        c += 1
      }
    }

    new DataSet(input, labels)
  }

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException("Not implemented")

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException("Not implemented")

  override def totalOutcomes(): Int = validCharacters.length

  override def getLabels: util.List[String] = throw new UnsupportedOperationException("Not implemented")

  override def inputColumns(): Int = validCharacters.length

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = miniBatchSize

  override def reset(): Unit = {
    exampleStartOffsets.clear
    initializeOffsets()
  }

  private def initializeOffsets(): Unit = {
    val nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2
    (0 until nMinibatchesPerEpoch).foreach { i =>
      exampleStartOffsets = exampleStartOffsets.+=(i * exampleLength)
    }
    exampleStartOffsets = rng.shuffle(exampleStartOffsets)
  }

  override def totalExamples(): Int = (fileCharacters.length - 1) / miniBatchSize - 2

  override def numExamples(): Int = totalExamples()

  override def next(): DataSet = next(miniBatchSize)

  override def hasNext: Boolean = exampleStartOffsets.nonEmpty
}

object CharacterIterator {
  /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
  def getMinimalCharacterSet: Array[Char] ={
    var validChars = new mutable.Queue[Char]()
    ('a' to 'z').foreach(c => validChars = validChars.+=(c))
    ('A' to 'Z').foreach(c => validChars = validChars.+=(c))
    ('0' to '9').foreach(c => validChars = validChars.+=(c))

    val temp = Array[Char]('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
    for(c <- temp) validChars = validChars.+=(c)

    val out = validChars.toArray
    out
  }

  def getDefaultCharacterSet: Array[Char] ={
    var validChars = new mutable.Queue[Char]()
    for (c <- getMinimalCharacterSet) {
      validChars = validChars.+=(c)
    }
    val additionalChars = Array[Char]('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
      '\\', '|', '<', '>')

    for (c <- additionalChars) {
      validChars = validChars.+=(c)
    }

    validChars.toArray
  }
}
