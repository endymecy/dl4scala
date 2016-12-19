package org.dl4scala.datasets.fetchers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import java.util

import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by endy on 16-12-19.
  */
abstract class BaseDataFetcher extends DataSetFetcher{

  protected val log: Logger = LoggerFactory.getLogger(classOf[BaseDataFetcher])

  var cursorVar: Int = 0
  var numOutcomesVar: Int = -1
  var inputColumnsVar: Int = -1
  var currVar: DataSet = _
  var totalExamplesVar: Int = 0

  /**
    * Creates a feature vector
    *
    * @param numRows the number of examples
    * @return a feature vector
    */
  protected def createInputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, inputColumnsVar)

  /**
    * Creates an output label matrix
    *
    * @param outcomeLabel the outcome label to use
    * @return a binary vector where 1 is transform to the
    *         index specified by outcomeLabel
    */
  protected def createOutputVector(outcomeLabel: Int): INDArray =
      FeatureUtil.toOutcomeVector(outcomeLabel, numOutcomesVar)

  protected def createOutputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, numOutcomesVar)

  /**
    * Initializes this data transform fetcher from the passed in datasets
    *
    * @param examples the examples to use
    */
  protected def initializeCurrFromList(examples: util.List[DataSet]) {
    if (examples.isEmpty) log.warn("Warning: empty dataset from the fetcher")
    //currVar = _
    val inputs: INDArray = createInputMatrix(examples.size)
    val labels: INDArray = createOutputMatrix(examples.size)

    0.until(examples.size).foreach{i =>
      val data: INDArray = examples.get(i).getFeatureMatrix
      val label: INDArray = examples.get(i).getLabels
      inputs.putRow(i, data)
      labels.putRow(i, label)
    }

    currVar = new DataSet(inputs, labels)
    examples.clear()
  }

  /**
    * Sets a list of label names to the curr dataset
    */
  def setLabelNames(names: util.List[String]) {
    currVar.setLabelNames(names)
  }

  def getLabelName(i: Int): String = currVar.getLabelNames.get(i)

  override def hasMore: Boolean = cursorVar < totalExamplesVar

  override def next: DataSet = currVar

  override def totalOutcomes: Int = numOutcomesVar

  override def inputColumns: Int = inputColumnsVar

  override def totalExamples: Int = totalExamplesVar

  override def reset() {
    cursorVar = 0
  }

  override def cursor: Int = {
    cursorVar
  }
}
