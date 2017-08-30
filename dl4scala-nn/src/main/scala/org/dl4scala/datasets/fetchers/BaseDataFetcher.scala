package org.dl4scala.datasets.fetchers

import org.dl4scala.datasets.iterator.DataSetFetcher
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/8/30.
  */
abstract class BaseDataFetcher extends DataSetFetcher{

  private val serialVersionUID = -859588773699432365L
  protected var cursorValue = 0
  protected var numOutcomes: Int = -1
  protected var inputColumnsValue: Int = -1
  protected var curr: DataSet = _
  protected var totalExamplesValue = 0
  protected val log: Logger = LoggerFactory.getLogger(classOf[BaseDataFetcher])

  /**
    * Creates a feature vector
    *
    * @param numRows the number of examples
    * @return a feature vector
    */
  protected def createInputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, inputColumnsValue)

  /**
    * Creates an output label matrix
    *
    * @param outcomeLabel the outcome label to use
    * @return a binary vector where 1 is transform to the
    *         index specified by outcomeLabel
    */
  protected def createOutputVector(outcomeLabel: Int): INDArray = FeatureUtil.toOutcomeVector(outcomeLabel, numOutcomes)

  protected def createOutputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, numOutcomes)


  /**
    * Initializes this data transform fetcher from the passed in datasets
    *
    * @param examples the examples to use
    */
  protected def initializeCurrFromList(examples: ArrayBuffer[DataSet]): Unit = {
    if (examples.isEmpty) log.warn("Warning: empty dataset from the fetcher")
    curr = null
    val inputs = createInputMatrix(examples.size)
    val labels = createOutputMatrix(examples.size)

    examples.indices.foreach{i =>
      val data: INDArray = examples(i).getFeatureMatrix
      val label: INDArray = examples(i).getLabels
      inputs.putRow(i, data)
      labels.putRow(i, label)
    }

    curr = new DataSet(inputs, labels)
    examples.clear()
  }

  /**
    * Sets a list of label names to the curr dataset
    */
  def setLabelNames(names: ArrayBuffer[String]): Unit = {
    curr.setLabelNames(names.asJava)
  }

  def getLabelName(i: Int): String = curr.getLabelNamesList.get(i)

  /**
    * Whether the dataset has more to load
    *
    * @return whether the data applyTransformToDestination has more to load
    */
  override def hasMore(): Boolean = cursorValue < totalExamplesValue

  /**
    * Returns the next data applyTransformToDestination
    *
    * @return the next dataset
    */
  override def next(): DataSet = curr

  /**
    * The number of labels for a dataset
    *
    * @return the number of labels for a dataset
    */
  override def totalOutcomes(): Int = numOutcomes
  /**
    * The length of a feature vector for an individual example
    *
    * @return the length of a feature vector for an individual example
    */
  override def inputColumns(): Int = inputColumnsValue

  /**
    * The total number of examples
    *
    * @return the total number of examples
    */
  override def totalExamples(): Int = totalExamplesValue

  /**
    * Returns the fetcher back to the beginning of the dataset
    */
  override def reset(): Unit = cursorValue = 0

  /**
    * Direct access to a number represenative of iterating through a dataset
    *
    * @return a cursor similar to an index
    */
  override def cursor(): Int = cursorValue
}
