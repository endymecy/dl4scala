package org.dl4scala.examples.transferlearning.vgg16

import java.io.File

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.{TrainedModelHelper, TrainedModels}
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.dl4scala.examples.transferlearning.vgg16.dataHelpers.FlowerDataSetIterator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by endy on 2017/6/25.
  */
object EditAtBottleneckOthersFrozen {
  private val log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckOthersFrozen.getClass)

  protected val numClasses = 5

  protected val seed = 12345
  private val trainPerc = 80
  private val batchSize = 15
  private val featureExtractionLayer = "block5_pool"

  def main(args: Array[String]): Unit = {
    // Import vgg
    //  Note that the model imported does not have an output layer (check printed summary)
    //  nor any training related configs (model from keras was imported with only weights and json)
    val modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16)
    val vgg16 = modelImportHelper.loadModel
    log.info(vgg16.summary)

    //Decide on a fine tune configuration to use.
    //In cases where there already exists a setting the fine tune setting will
    //  override the setting for all layers that are not "frozen".
    val fineTuneConf = new FineTuneConfiguration.Builder()
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.RELU)
      .learningRate(5e-5)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .dropOut(0.5)
      .seed(seed)
      .build()

    //Construct a new model with the intended architecture and print summary
    //  Note: This architecture is constructed with the primary intent of demonstrating use of the transfer learning API,
    //        secondary to what might give better results
    val vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
      .fineTuneConfiguration(fineTuneConf)
      .setFeatureExtractor(featureExtractionLayer) //"block5_pool" and below are frozen
      .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
      .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
      .addLayer("fc3",new DenseLayer.Builder().activation(Activation.TANH).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
      .addLayer("newpredictions",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation(Activation.SOFTMAX)
          .nIn(256)
          .nOut(numClasses)
          .build(),"fc3") //add in a final output dense layer,
            // note that learning related configurations applied on a new layer here will be honored
            // In other words - these will override the finetune confs.
           // For eg. activation function will be softmax not RELU
      .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
      .build()

    log.info(vgg16Transfer.summary)

    //Dataset iterators
    FlowerDataSetIterator.setup(batchSize, trainPerc)
    val trainIter: DataSetIterator = FlowerDataSetIterator.trainIterator
    val testIter: DataSetIterator = FlowerDataSetIterator.testIterator

    var eval: Evaluation = null
    eval = vgg16Transfer.evaluate(testIter)
    log.info("Eval stats BEFORE fit.....")
    log.info(eval.stats + "\n")

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

    //Save the model

    //Note that the saved model will not know which layers were frozen during training.
    //Frozen models always have to specified before training.
    // Models with frozen layers can be constructed in the following two ways:
    //      1. .setFeatureExtractor in the transfer learning API which will always a return a new model (as seen in this example)
    //      2. in place with the TransferLearningHelper constructor which will take a model, and a specific vertexname
    //              and freeze it and the vertices on the path from an input to it (as seen in the FeaturizePreSave class)
    //The saved model can be "fine-tuned" further as in the class "FitFromFeaturized"
    val locationToSave: File = new File("MyComputationGraph.zip")
    val saveUpdater: Boolean = false
    ModelSerializer.writeModel(vgg16Transfer, locationToSave, saveUpdater)
    log.info("Model saved")
  }
}
