package org.dl4scala.examples.recurrent.seq2seq

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.{MultiDataSet, MultiDataSetPreProcessor}
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable
import scala.util.Random
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/10.
  */
class CustomSequenceIterator(seed: Int, batchSize: Int, totalBatches: Int) extends MultiDataSetIterator{
  import CustomSequenceIterator._

  private var randnumG = new Random(seed)
  private var toTestSet = false
  private var currentBatch = 0
  private var seenSequences = new mutable.HashSet[String]()

  oneHotEncoding()

  def generateTest(testSize: Int): MultiDataSet = {
    toTestSet = true
    val testData = next(testSize)
    reset()
    testData
  }

  override def next(sampleSize: Int): MultiDataSet = {
    var currentCount = 0
    var num1 = 0
    var num2 = 0
    val encoderSeqList = new mutable.ArrayBuffer[INDArray]()
    val decoderSeqList = new mutable.ArrayBuffer[INDArray]()
    val outputSeqList = new mutable.ArrayBuffer[INDArray]()

    while (currentCount < sampleSize) {
      var break = false
      while(break) {
        num1 = randnumG.nextInt(Math.pow(10, numDigits).toInt)
        num2 = randnumG.nextInt(Math.pow(10, numDigits).toInt)
        val forSum = String.valueOf(num1) + "+" + String.valueOf(num2)
        if (seenSequences.add(forSum)) break = true
      }
      val encoderInput = prepToString(num1, num2)
      encoderSeqList.append(mapToOneHot(encoderInput))
      val decoderInput = prepToString(num1 + num2, goFirst = true)

      if (toTestSet) {
        //wipe out everything after "go"; not necessary since we do not use these at test time but here for clarity
        (1 until decoderInput.length).foreach(i => decoderInput(i) = " ")
      }
      decoderSeqList.append(mapToOneHot(decoderInput))
      val decoderOutput: Array[String] = prepToString(num1 + num2, goFirst = false)
      outputSeqList.append(mapToOneHot(decoderOutput))
      currentCount += 1
    }

    val encoderSeq = Nd4j.vstack(encoderSeqList.asJava)
    val decoderSeq = Nd4j.vstack(decoderSeqList.asJava)
    val outputSeq = Nd4j.vstack(outputSeqList.asJava)

    val inputs = Array[INDArray](encoderSeq, decoderSeq)
    val inputMasks = Array[INDArray](Nd4j.ones(sampleSize, numDigits * 2 + 1), Nd4j.ones(sampleSize, numDigits + 1 + 1))
    val labels = Array[INDArray](outputSeq)
    val labelMasks = Array[INDArray](Nd4j.ones(sampleSize, numDigits + 1 + 1))

    currentBatch += 1

    new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks)
  }

  override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = _

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = false

  override def reset(): Unit = {
    currentBatch = 0
    toTestSet = false
    seenSequences = new mutable.HashSet[String]()
    randnumG = new Random(seed)
  }

  override def next(): MultiDataSet = next(batchSize)

  override def hasNext: Boolean = currentBatch < totalBatches


  // Helper method for encoder input
  // Given two numbers, num1 and num, returns a string array which represents the input to the encoder RNN
  // Note that the string is padded to the correct length and reversed
  // Eg. num1 = 7, num 2 = 13 will return {"3","1","+","7"," "}
  def prepToString(num1: Int, num2: Int): Array[String] = {
    val encoded = new Array[String](numDigits * 2 + 1)
    var num1S = String.valueOf(num1)
    var num2S = String.valueOf(num2)
    // padding
    while (num1S.length < numDigits) num1S = " " + num1S
    while (num2S.length < numDigits) num2S = " " + num2S

    val sumString = num1S + "+" + num2S

    encoded.indices.foreach{i =>
      encoded((encoded.length - 1) - i) = Character.toString(sumString.charAt(i))
    }

    encoded
  }

  // Helper method for decoder input when goFirst
  // for decoder output when !goFirst
  // Given a number, return a string array which represents the decoder input (or output) given goFirst (or !goFirst)
  // eg. For numDigits = 2 and sum = 31
  // if goFirst will return  {"go","3","1", " "}
  // if !goFirst will return {"3","1"," ","eos"}
  def prepToString(sum: Int, goFirst: Boolean): Array[String] = {
    var start = 0
    var end = 0
    val decoded = new Array[String](numDigits + 1 + 1)
    if (goFirst) {
      decoded(0) = "Go"
      start = 1
      end = decoded.length - 1
    } else {
      start = 0
      end = decoded.length - 2
      decoded(decoded.length - 1) = "End"
    }

    val sumString: String = String.valueOf(sum)
    var maxIndex = start
    // add in digits
    (0 until sumString.length).foreach{i =>
      decoded(start + i) = Character.toString(sumString.charAt(i))
      maxIndex += 1
    }

    // needed padding
    while (maxIndex <= end) {
      decoded(maxIndex) = " "
      maxIndex += 1
    }

    decoded
  }
}

object CustomSequenceIterator {
  val SEQ_VECTOR_DIM = 14
  private val numDigits = 2

  val oneHotMap = new mutable.OpenHashMap[String, Int]()
  val oneHotOrder = new Array[String](SEQ_VECTOR_DIM)

  // Takes in an array of strings and return a one hot encoded array of size 1 x 14 x timesteps
  // Each element in the array indicates a time step
  // Length of one hot vector = 14
  private def mapToOneHot(toEncode: Array[String]) = {
    val ret = Nd4j.zeros(1, SEQ_VECTOR_DIM, toEncode.length)
    toEncode.indices.foreach(i => ret.putScalar(0, oneHotMap.getOrElse(toEncode(i), 0), i, 1))
    ret
  }

  def mapToString(encodeSeq: INDArray, decodeSeq: INDArray): String = mapToString(encodeSeq, decodeSeq, " --> ")

  def mapToString( encodeSeq: INDArray,  decodeSeq: INDArray,  sep: String): String = {
    var ret = ""
    val encodeSeqS = oneHotDecode(encodeSeq)
    val decodeSeqS = oneHotDecode(decodeSeq)
    encodeSeqS.indices.foreach(i => ret += "\t" + encodeSeqS(i) + sep + decodeSeqS(i) + "\n")
    ret
  }

  // Helper method that takes in a one hot encoded INDArray and returns an interpreted array of strings
  // toInterpret size batchSize x one_hot_vector_size(14) x time_steps
  def oneHotDecode(toInterpret: INDArray): Array[String] = {
    val decodedString = new Array[String](toInterpret.size(0))
    val oneHotIndices = Nd4j.argMax(toInterpret, 1)
    // drops a dimension, so now a two dim array of shape batchSize x time_steps
    var i = 0
    while (i < oneHotIndices.size(0)) {
      val currentSlice = oneHotIndices.slice(i).dup.data.asInt //each slice is a batch
      decodedString(i) = mapFromOneHot(currentSlice)
      i += 1
    }
    decodedString
  }

  private def mapFromOneHot(toMap: Array[Int]): String = {
    var ret = ""
    toMap.indices.foreach{i =>
      ret += oneHotOrder(toMap(i))
    }
    if (toMap.length > numDigits + 1 + 1)  new StringBuilder(ret).reverse.toString
    else ret
  }

  // One hot encoding map
  private def oneHotEncoding() = {
    (0 until 10).foreach{i =>
      oneHotOrder(i) = String.valueOf(i)
      oneHotMap.put(String.valueOf(i), i)
    }

    oneHotOrder(10) = " "
    oneHotMap.put(" ", 10)

    oneHotOrder(11) = "+"
    oneHotMap.put("+", 11)

    oneHotOrder(12) = "Go"
    oneHotMap.put("Go", 12)

    oneHotOrder(13) = "End"
    oneHotMap.put("End", 13)
  }
}
