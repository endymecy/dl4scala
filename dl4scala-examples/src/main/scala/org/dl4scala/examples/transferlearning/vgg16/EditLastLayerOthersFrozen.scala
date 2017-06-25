package org.dl4scala.examples.transferlearning.vgg16

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning}
import org.deeplearning4j.nn.weights.WeightInit
import org.dl4scala.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by endy on 2017/6/25.
  */
object EditLastLayerOthersFrozen {
  private val log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.getClass)

  protected val numClasses = 5
  protected val seed = 12345

  private val trainPerc = 80
  private val batchSize = 15
  private val featureExtractionLayer = "fc2"

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
      .learningRate(5e-5)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .seed(seed)
      .build()

    //Construct a new model with the intended architecture and print summary
    val vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
      .fineTuneConfiguration(fineTuneConf)
      .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
      .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
      .addLayer("predictions",
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(4096).nOut(numClasses)
        .weightInit(WeightInit.DISTRIBUTION)
        .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
        .activation(Activation.SOFTMAX).build(),
      "fc2")
      .build()

    log.info(vgg16Transfer.summary)

    //Dataset iterators
    FlowerDataSetIterator.setup(batchSize, trainPerc)
    val trainIter = FlowerDataSetIterator.trainIterator
    val testIter = FlowerDataSetIterator.testIterator

    var eval: Evaluation = null
    eval = vgg16Transfer.evaluate(testIter)
    log.info("Eval stats BEFORE fit.....")
    log.info(eval.stats + "\n")

    testIter.reset()

    var iter = 0
    while (trainIter.hasNext) {
      vgg16Transfer.fit(trainIter.next())
      if (iter % 10 == 0) {
        log.info("Evaluate model at iter " + iter + " ....")
        eval = vgg16Transfer.evaluate(testIter)
        log.info(eval.stats)
        testIter.reset()
      }
      iter += 1
    }

    log.info("Model build complete")
  }
}
