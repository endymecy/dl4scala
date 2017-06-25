package org.dl4scala.examples.transferlearning.vgg16.dataHelpers

import java.io.File

import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper
import org.nd4j.linalg.dataset.DataSet

/**
  * The TransferLearningHelper class allows users to "featurize" a dataset at specific intermediate vertices/layers of a model
  * This example demonstrates how to presave these
  * Refer to the "FitFromFeaturized" example for how to fit a model with these featurized datasets
  *
  * Created by endy on 2017/6/25.
  */
object FeaturizedPreSave {

  private val log = org.slf4j.LoggerFactory.getLogger(FeaturizedPreSave.getClass)

  private val trainPerc = 80
  protected val batchSize = 15
  val featurizeExtractionLayer = "fc2"

  def main(args: Array[String]): Unit = {
    val modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16)
    val vgg16 = modelImportHelper.loadModel
    log.info(vgg16.summary)


    //use the TransferLearningHelper to freeze the specified vertices and below
    //NOTE: This is done in place! Pass in a cloned version of the model if you would prefer to not do this in place
    val transferLearningHelper = new TransferLearningHelper(vgg16, featurizeExtractionLayer)
    log.info(vgg16.summary)

    FlowerDataSetIterator.setup(batchSize, trainPerc)
    val trainIter = FlowerDataSetIterator.trainIterator
    val testIter = FlowerDataSetIterator.testIterator

    var trainDataSaved = 0
    while (trainIter.hasNext) {
      val currentFeaturized = transferLearningHelper.featurize(trainIter.next())
      saveToDisk(currentFeaturized, trainDataSaved, isTrain = true)
      trainDataSaved += 1
    }

    var testDataSaved = 0
    while (testIter.hasNext) {
      val currentFeaturized = transferLearningHelper.featurize(testIter.next())
      saveToDisk(currentFeaturized, testDataSaved, isTrain = false)
      testDataSaved += 1
    }

    log.info("Finished pre saving featurized test and train data")
  }

  def saveToDisk(currentFeaturized: DataSet, iterNum: Int, isTrain: Boolean): Unit = {
    val fileFolder = if (isTrain) new File("trainFolder")
    else new File("testFolder")

    if (iterNum == 0) fileFolder.mkdirs
    var fileName = "flowers-" + featurizeExtractionLayer + "-"
    if (isTrain) fileName +=  "train-" else fileName +=  "test-"
    fileName += iterNum + ".bin"
    currentFeaturized.save(new File(fileFolder, fileName))
    log.info("Saved " + (if (isTrain) "train "
    else "test ") + "dataset #" + iterNum)
  }
}
