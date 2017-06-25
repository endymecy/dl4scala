package org.dl4scala.examples.transferlearning.vgg16.dataHelpers

import java.io.File

/**
  * Created by endy on 2017/6/25.
  */
object FlowerDataSetIteratorFeaturized {

  private val log = org.slf4j.LoggerFactory.getLogger(FlowerDataSetIteratorFeaturized.getClass)

  var featureExtractorLayer: String = FeaturizedPreSave.featurizeExtractionLayer

  def setup(featureExtractorLayerArg: String): Unit = {
    featureExtractorLayer = featureExtractorLayerArg
  }

  import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
  import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException
  import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException
  import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator
  import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
  import java.io.IOException

  @throws(classOf[UnsupportedKerasConfigurationException])
  @throws(classOf[IOException])
  @throws(classOf[InvalidKerasConfigurationException])
  def trainIterator: DataSetIterator = {
    runFeaturize()
    val existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"), "flowers-" + featureExtractorLayer + "-train-%d.bin")
    val asyncTrainIter = new AsyncDataSetIterator(existingTrainingData)
    asyncTrainIter
  }

  def testIterator: DataSetIterator = {
    val existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"), "flowers-" + featureExtractorLayer + "-test-%d.bin")
    val asyncTestIter = new AsyncDataSetIterator(existingTestData)
    asyncTestIter
  }

  @throws(classOf[UnsupportedKerasConfigurationException])
  @throws(classOf[IOException])
  @throws(classOf[InvalidKerasConfigurationException])
  private def runFeaturize() = {
    val trainDir = new File("trainFolder", "flowers-" + featureExtractorLayer + "-train-0.bin")
    if (!trainDir.isFile) {
      log.info("\n\tFEATURIZED DATA NOT FOUND. \n\t\tRUNNING \"FeaturizedPreSave\" first to do presave of featurized data")
      FeaturizedPreSave.main(null)
    }
  }
}
