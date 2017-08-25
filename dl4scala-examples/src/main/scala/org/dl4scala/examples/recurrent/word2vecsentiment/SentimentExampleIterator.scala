package org.dl4scala.examples.recurrent.word2vecsentiment

import java.io.{File, IOException}

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/5/26.
  */
class SentimentExampleIterator(dataDirectory: String, wordVectors: WordVectors, batchSize: Int,
                               truncateLength: Int, train: Boolean) extends DataSetIterator{

  private val vectorSize = wordVectors.getWordVector(wordVectors.vocab.wordAtIndex(0)).length
  private val p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/pos/") + "/")
  private val n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/neg/") + "/")
  private val positiveFiles: Array[File] = p.listFiles()
  private val negativeFiles: Array[File] = n.listFiles()
  private val tokenizerFactory = new DefaultTokenizerFactory()
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)
  private var cursor_int = 0

  override def cursor(): Int = cursor_int

  override def next(num: Int): DataSet = {
    if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException
    try
       nextDataSet(num)
    catch {
      case e: IOException =>
        throw new RuntimeException(e)
    }
  }

  @throws(classOf[IOException])
  private def nextDataSet(num: Int): DataSet = {
    // First: load reviews to String. Alternate positive and negative reviews
    val reviews = new ArrayBuffer[String](num)
    val positive = new Array[Boolean](num)
    for(i <- 0 until num if cursor_int < totalExamples) {
      if(cursor % 2 == 0){
        //Load positive review
        val posReviewNumber: Int = cursor / 2
        val review: String = FileUtils.readFileToString(positiveFiles(posReviewNumber))
        reviews.append(review)
        positive(i) = true
      } else {
        //Load negative review
        val negReviewNumber: Int = cursor / 2
        val review: String = FileUtils.readFileToString(negativeFiles(negReviewNumber))
        reviews.append(review)
        positive(i) = true
      }
      cursor_int = cursor_int + 1
    }

    // Second: tokenize reviews and filter out unknown words
    val allTokens = new ArrayBuffer[ArrayBuffer[String]](reviews.size)
    var maxLength = 0
    for(s: String <- reviews){
      val tokens = tokenizerFactory.create(s).getTokens
      val tokensFiltered = new ArrayBuffer[String]
      for(token <- tokens.asScala){
        if (wordVectors.hasWord(token)) tokensFiltered.append(token)
      }
      allTokens.append(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }

    // If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    if(maxLength > truncateLength) maxLength = truncateLength

    // Create data for training
    // Here: we have reviews.size() examples of varying lengths
//    val features = Nd4j.create(reviews.size, vectorSize, maxLength)
//    val labels = Nd4j.create(reviews.size, 2, maxLength) //Two labels: positive or negative

    val features = Nd4j.create(Array[Int](reviews.size, vectorSize, maxLength), 'f')
    val labels = Nd4j.create(Array[Int](reviews.size, 2, maxLength), 'f')

    // Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
    // Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    val featuresMask = Nd4j.zeros(reviews.size, maxLength)
    val labelsMask = Nd4j.zeros(reviews.size, maxLength)

    val temp = new Array[Int](2)
    for(i <- reviews.indices){
      val tokens = allTokens(i)
      temp(0) = i
      for(j <- tokens.indices if j < maxLength) {
        val token = tokens(j)
        val vector = wordVectors.getWordVectorMatrix(token)
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.point(j)), vector)
        temp(1) = j
        // Word is present (not padding) for this example + time step -> 1.0 in features mask
        featuresMask.putScalar(temp, 1.0)
      }

      val idx = if (positive(i)) 0 else 1
      val lastIdx = Math.min(tokens.size, maxLength)
      // Set label: [0,1] for negative, [1,0] for positive
      labels.putScalar(Array[Int](i, idx, lastIdx - 1), 1.0)
      // Specify that an output exists at the final time step for this example
      labelsMask.putScalar(Array[Int](i, lastIdx - 1), 1.0)
    }

    new DataSet(features, labels, featuresMask, labelsMask)
  }

  override def setPreProcessor(dataSetPreProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException("Not implemented")

  override def totalOutcomes(): Int = 2

  override def getLabels: java.util.List[String] = ArrayBuffer("positive", "negative").asJava

  override def inputColumns(): Int = vectorSize

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = batchSize

  override def reset(): Unit = {
    cursor_int = 0
  }

  override def totalExamples(): Int = positiveFiles.length + negativeFiles.length

  override def numExamples(): Int = totalExamples()

  override def next(): DataSet = next(batchSize)

  override def hasNext: Boolean = cursor_int < numExamples


  /**
    * Used post training to load a review from a file to a features INDArray that can be passed to the network output method
    *
    * @param file      File to load the review from
    * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array
    * @throws IOException If file cannot be read
    */
  @throws(classOf[IOException])
  def loadFeaturesFromFile(file: File, maxLength: Int): INDArray = {
    val review = FileUtils.readFileToString(file)
    loadFeaturesFromString(review, maxLength)
  }

  /**
    * Used post training to convert a String to a features INDArray that can be passed to the network output method
    *
    * @param reviewContents Contents of the review to vectorize
    * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array for the given input String
    */
  def loadFeaturesFromString(reviewContents: String, maxLength: Int): INDArray = {
    val tokens = tokenizerFactory.create(reviewContents).getTokens
    val tokensFiltered = new ArrayBuffer[String]

    for(token <- tokens.asScala){
      if (wordVectors.hasWord(token)) tokensFiltered.append(token)
    }
    val outputLength = Math.max(maxLength, tokensFiltered.size)
    val features = Nd4j.create(1, vectorSize, outputLength)

    for(j <- 0 until tokens.size if j < maxLength) {
      val token = tokens.get(j)
      val vector = wordVectors.getWordVectorMatrix(token)
      features.put(Array[INDArrayIndex](NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(j)), vector)
    }

    features
  }
}
