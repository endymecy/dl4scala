package org.dl4scala.datasets.fetchers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import java.util

import com.typesafe.scalalogging.LazyLogging

/**
  * Created by endy on 16-12-19.
  */
abstract class BaseDataFetcher(var cursor: Int = 0,
                               var inputColumns: Int = -1,
                               var totalExamples: Int = 0,
                               var curr: DataSet = null,
                               var numOutcomes: Int = -1) extends DataSetFetcher with LazyLogging{

  /**
    * Creates a feature vector
    *
    * @param numRows the number of examples
    * @return a feature vector
    */
  protected def createInputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, inputColumns)

  /**
    * Creates an output label matrix
    *
    * @param outcomeLabel the outcome label to use
    * @return a binary vector where 1 is transform to the
    *         index specified by outcomeLabel
    */
  protected def createOutputVector(outcomeLabel: Int): INDArray =
      FeatureUtil.toOutcomeVector(outcomeLabel, numOutcomes)

  protected def createOutputMatrix(numRows: Int): INDArray = Nd4j.create(numRows, numOutcomes)

  /**
    * Initializes this data transform fetcher from the passed in datasets
    *
    * @param examples the examples to use
    */
  protected def initializeCurrFromList(examples: util.List[DataSet]) {
    if (examples.isEmpty) logger.warn("Warning: empty dataset from the fetcher")
    curr = null
    val inputs: INDArray = createInputMatrix(examples.size)
    val labels: INDArray = createOutputMatrix(examples.size)

    0.until(examples.size).foreach{i =>
      val data: INDArray = examples.get(i).getFeatureMatrix
      val label: INDArray = examples.get(i).getLabels
      inputs.putRow(i, data)
      labels.putRow(i, label)
    }

    curr = new DataSet(inputs, labels)
    examples.clear()
  }

  /**
    * Sets a list of label names to the curr dataset
    */
  def setLabelNames(names: util.List[String]) {
    curr.setLabelNames(names)
  }

  def getLabelName(i: Int): String = curr.getLabelNamesList.get(i)

  override def hasMore: Boolean = cursor < totalExamples

  override def next: DataSet = curr

  override def totalOutcomes: Int = numOutcomes

  override def reset() {
    cursor = 0
  }
}
