package org.dl4scala.examples.recurrent.processnews

import java.io.{BufferedReader, File, FileReader, IOException}
import java.util

import org.apache.commons.io.FileUtils
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/4.
  */
class NewsIterator(dataDirectory: String,
                   wordVectors: WordVectors,
                   batchSize: Int,
                   truncateLength: Int,
                   train: Boolean,
                   tokenizerFactory: TokenizerFactory) extends DataSetIterator{
  private val logger: Logger = LoggerFactory.getLogger(classOf[NewsIterator])

  private val vectorSize = wordVectors.getWordVector(wordVectors.vocab.wordAtIndex(0)).length

  private var totalNews = 0

  private val categoryData = new ArrayBuffer[(String, ArrayBuffer[String])]
  populateData(train)

  private val labels = {
    val arr = new ArrayBuffer[String]()
    categoryData.indices.foreach{i =>
      arr.append(this.categoryData(i)._1.split(",")(1))
    }
    arr
  }
  private var cursor_val = 0
  private var newsPosition = 0
  private var currCategory = 0
  private var maxLength = 0


  // This function loads news headlines from files stored in resources into categoryData List.
  private def populateData(train: Boolean) = {
    val categories = new File(this.dataDirectory + File.separator + "categories.txt")
    try {
      val brCategories = new BufferedReader(new FileReader(categories))
      Stream.continually(brCategories.readLine()).takeWhile(line => line != null).foreach{line =>
        val curFileName =
          if(train) this.dataDirectory + File.separator + "train" + File.separator + line.split(",")(0) + ".txt"
          else this.dataDirectory + File.separator + "test" + File.separator + line.split(",")(0) + ".txt"

        val currFile = new File(curFileName)
        val currBR = new BufferedReader(new FileReader(currFile))
        val tempList = new ArrayBuffer[String]()
        Stream.continually(currBR.readLine()).takeWhile(tempCurrLine => tempCurrLine != null).foreach{tempCurrLine =>
          tempList.append(tempCurrLine)
          this.totalNews += 1
        }
        currBR.close()
        this.categoryData.append((line, tempList))
      }
      brCategories.close()
    } catch {
      case e: Exception =>
        logger.info("Exception in reading file :" + e.getMessage)
    }
  }


  override def cursor(): Int = this.cursor_val

  override def next(num: Int): DataSet = {
    if (this.cursor_val >= this.totalNews) throw new NoSuchElementException
    try
      nextDataSet(num)
    catch {
      case e: IOException =>
        throw new RuntimeException(e)
    }
  }

  override def setPreProcessor(dataSetPreProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException()

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException("Not implemented")

  override def totalOutcomes(): Int = this.categoryData.size

  override def getLabels: util.List[String] = this.labels.asJava

  override def inputColumns(): Int = this.vectorSize

  override def resetSupported(): Boolean = true

  override def asyncSupported(): Boolean = true

  override def batch(): Int = this.batchSize

  override def reset(): Unit = {
    this.cursor_val = 0
    this.newsPosition = 0
    this.currCategory = 0
  }

  override def totalExamples(): Int = this.totalNews

  override def numExamples(): Int = totalExamples()

  override def next(): DataSet = next(batchSize)

  override def hasNext: Boolean = this.cursor_val < numExamples()

  @throws(classOf[IOException])
  private def nextDataSet(num: Int): DataSet = {
    // Loads news into news list from categoryData List along with category of each news
    val news = new ArrayBuffer[String](num)
    val category = new Array[Int](num)

    var i = 0
    while(i < num && cursor_val < totalExamples()) {
      if (currCategory < categoryData.size) {
        news.append(this.categoryData(currCategory)._2(newsPosition))
        category(i) = this.categoryData(currCategory)._1.split(",")(0).toInt
        currCategory += 1
        cursor_val += 1
      } else {
        currCategory = 0
        newsPosition += 1
        i -= 1
      }
      i += 1
    }
    // Second: tokenize news and filter out unknown words
    val allTokens = new ArrayBuffer[ArrayBuffer[String]](news.size)
    maxLength = 0
    for(s <- news){
      val tokens = tokenizerFactory.create(s).getTokens
      val tokensFiltered = new ArrayBuffer[String]
      for (t <- tokens) {
        if (wordVectors.hasWord(t)) tokensFiltered.append(t)
      }
      allTokens.append(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }

    // If longest news exceeds 'truncateLength': only take the first 'truncateLength' words
    if (maxLength > truncateLength) maxLength = truncateLength

    // Create data for training
    // Here: we have news.size() examples of varying lengths
    val features = Nd4j.create(news.size, vectorSize, maxLength)
    val labels = Nd4j.create(news.size, this.categoryData.size, maxLength) //Three labels: Crime, Politics, Bollywood

    // Because we are dealing with news of different lengths and only one output at the final time step: use padding arrays
    // Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    val featuresMask = Nd4j.zeros(news.size, maxLength)
    val labelsMask = Nd4j.zeros(news.size, maxLength)

    val temp = new Array[Int](2)

    news.indices.foreach{i =>
      val tokens = allTokens(i)
      temp(0) = i
      for(j <- tokens.indices if j < maxLength) {
        val token = tokens(j)
        val vector = wordVectors.getWordVectorMatrix(token)
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)), vector)
        temp(1) = j
        featuresMask.putScalar(temp, 1.0)
      }
      val idx = category(i)
      val lastIdx = Math.min(tokens.size, maxLength)
      labels.putScalar(Array[Int](i, idx, lastIdx - 1), 1.0)
      labelsMask.putScalar(Array[Int](i, lastIdx - 1), 1.0)
    }
    new DataSet(features, labels, featuresMask, labelsMask)
  }

  /**
    * Used post training to load a review from a file to a features INDArray that can be passed to the network output method
    *
    * @param file      File to load the review from
    * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array
    * @throws IOException If file cannot be read
    */
  @throws(classOf[IOException])
  def loadFeaturesFromFile(file: Nothing, maxLength: Int): INDArray = {
    val news = FileUtils.readFileToString(file)
    loadFeaturesFromString(news, maxLength)
  }

  /**
    * Used post training to convert a String to a features INDArray that can be passed to the network output method
    *
    * @param reviewContents Contents of the review to vectorize
    * @param maxLength      Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array for the given input String
    */
  def loadFeaturesFromString(reviewContents: String, maxLength: Int): INDArray = {
    val tokens = tokenizerFactory.create(reviewContents).getTokens
    val tokensFiltered = new ArrayBuffer[String]
    for (t <- tokens) {
      if (wordVectors.hasWord(t)) tokensFiltered.append(t)
    }
    val outputLength = Math.max(maxLength, tokensFiltered.size)

    val features = Nd4j.create(1, vectorSize, outputLength)

    for(j <- 0 until tokens.size() if j < maxLength){
      val token = tokens.get(j)
      val vector = wordVectors.getWordVectorMatrix(token)
      features.put(Array[INDArrayIndex](NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)), vector)
    }
    features
  }
}

object NewsIterator {
  class Builder() {
    private var dataDirectory: String = _
    private var wordVectors: WordVectors = _
    private var batchSize = 0
    private var truncateLength = 0
    var tokenizerFactory: TokenizerFactory = _
    private var train: Boolean = _

    def dataDirectory(dataDirectory: String): NewsIterator.Builder = {
      this.dataDirectory = dataDirectory
      this
    }

    def wordVectors(wordVectors: WordVectors): NewsIterator.Builder = {
      this.wordVectors = wordVectors
      this
    }

    def batchSize(batchSize: Int): NewsIterator.Builder = {
      this.batchSize = batchSize
      this
    }

    def truncateLength(truncateLength: Int): NewsIterator.Builder = {
      this.truncateLength = truncateLength
      this
    }

    def train(train: Boolean): NewsIterator.Builder = {
      this.train = train
      this
    }

    def tokenizerFactory(tokenizerFactory: TokenizerFactory): NewsIterator.Builder = {
      this.tokenizerFactory = tokenizerFactory
      this
    }

    def build() = new NewsIterator(dataDirectory, wordVectors, batchSize, truncateLength, train, tokenizerFactory)

    override def toString: String = "org.dl4scala.examples.recurrent.ProcessNews.NewsIterator.Builder(dataDirectory=" +
      this.dataDirectory + ", wordVectors=" + this.wordVectors + ", batchSize=" + this.batchSize +
      ", truncateLength=" + this.truncateLength + ", train=" + this.train + ")"
  }
}