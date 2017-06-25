package org.dl4scala.examples.transferlearning.vgg16

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning, TransferLearningHelper}
import org.deeplearning4j.nn.weights.WeightInit
import org.dl4scala.examples.transferlearning.vgg16.dataHelpers.{FeaturizedPreSave, FlowerDataSetIteratorFeaturized}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by endy on 2017/6/25.
  */
object FitFromFeaturized {
  private val log = org.slf4j.LoggerFactory.getLogger(FitFromFeaturized.getClass)

  val featureExtractionLayer: String = FeaturizedPreSave.featurizeExtractionLayer
  protected val seed = 12345
  protected val numClasses = 5
  protected val nEpochs = 3

  def main(args: Array[String]): Unit = {
    //Import vgg
    //Note that the model imported does not have an output layer (check printed summary)
    //  nor any training related configs (model from keras was imported with only weights and json)
    val modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16)
    val vgg16 = modelImportHelper.loadModel
    log.info(vgg16.summary)

    //Decide on a fine tune configuration to use.
    //In cases where there already exists a setting the fine tune setting will
    //  override the setting for all layers that are not "frozen".
    val fineTuneConf = new FineTuneConfiguration.Builder()
      .learningRate(3e-5)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .seed(seed)
      .build()


    //Construct a new model with the intended architecture and print summary

    val vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
      .fineTuneConfiguration(fineTuneConf)
      .setFeatureExtractor(featureExtractionLayer)
      .removeVertexKeepConnections("predictions") //the specified layer and below are "frozen"
      .addLayer("predictions",
          new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(4096)
            .nOut(numClasses)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numClasses))))
            .activation(Activation.SOFTMAX).build, "fc2")
      .build
    log.info(vgg16Transfer.summary)

    val trainIter = FlowerDataSetIteratorFeaturized.trainIterator
    val testIter = FlowerDataSetIteratorFeaturized.testIterator

    //Instantiate the transfer learning helper to fit and output from the featurized dataset

    //The .unfrozenGraph() is the unfrozen subset of the computation graph passed in.
    //If using with a UI or a listener attach them directly to the unfrozenGraph instance
    //With each iteration updated params from unfrozenGraph are copied over to the original model
    val transferLearningHelper = new TransferLearningHelper(vgg16Transfer)
    log.info(transferLearningHelper.unfrozenGraph.summary)

    (0 until nEpochs).foreach{epoch =>
      if (epoch == 0) {
        val eval = transferLearningHelper.unfrozenGraph.evaluate(testIter)
        log.info("Eval stats BEFORE fit.....")
        log.info(eval.stats + "\n")
        testIter.reset()
      }

      var iter = 0
      while (trainIter.hasNext) {
        transferLearningHelper.fitFeaturized(trainIter.next())
        if (iter % 10 == 0) {
          log.info("Evaluate model at iter " + iter + " ....")
          val eval = transferLearningHelper.unfrozenGraph.evaluate(testIter)
          log.info(eval.stats)
          testIter.reset()
        }
        iter += 1
      }
      trainIter.reset()
      log.info("Epoch #" + epoch + " complete")
    }
    log.info("Model build complete")
  }
}
