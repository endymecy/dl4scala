package org.dl4scala.examples.misc.centerloss

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{CenterLossOutputLayer, ConvolutionLayer, DenseLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.dl4scala.examples.unsupervised.variational.plot.PlotUtil
import org.dl4scala.examples.userInterface.util.GradientsAndParamsListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

/**
  * Example: training an embedding using the center loss model, on MNIST
  * The motivation is to use the class labels to learn embeddings that have the following properties:
  * (a) Intra-class similarity (i.e., similar vectors for same numbers)
  * (b) Inter-class dissimilarity (i.e., different vectors for different numbers)
  *
  * Refer to the paper "A Discriminative Feature Learning Approach for Deep Face Recognition", Wen et al. (2016)
  * http://ydwen.github.io/papers/WenECCV16.pdf
  *
  * Created by endy on 2017/6/25.
  */
object CenterLossLenetMnistExample {
  private val log = LoggerFactory.getLogger(CenterLossLenetMnistExample.getClass)

  def main(args: Array[String]): Unit = {
    val outputNum = 10
    // The number of possible outcomes
    val batchSize = 64
    // Test batch size
    val nEpochs = 10
    // Number of training epochs
    val seed = 123

    //Lambda defines the relative strength of the center loss component.
    //lambda = 0.0 is equivalent to training with standard softmax only
    val lambda = 1.0

    //Alpha can be thought of as the learning rate for the centers for each class
    val alpha = 0.1

    log.info("Load data....")
    val mnistTrain = new MnistDataSetIterator(batchSize, true, 12345)
    val mnistTest = new MnistDataSetIterator(10000, false, 12345)


    log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .regularization(true)
      .l2(0.0005)
      .learningRate(0.01)
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.RELU)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAM)
      .adamMeanDecay(0.9)
      .adamVarDecay(0.999)
      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(32).activation(Activation.LEAKYRELU).build)
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(64).build)
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build)
      .layer(4, new DenseLayer.Builder().nOut(256).build)
      .layer(5, new DenseLayer.Builder().activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).nOut(2).l2(0.1).build)
      .layer(6, new CenterLossOutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(2).nOut(outputNum).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).alpha(alpha).lambda(lambda).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1))
      .backprop(true)
      .pretrain(false)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()

    log.info("Train model....")
    model.setListeners(new GradientsAndParamsListener(model, 100), new ScoreIterationListener(100))

    val embeddingByEpoch = new ArrayBuffer[(INDArray, INDArray)]
    val epochNum = new ArrayBuffer[Int]

    val testData = mnistTest.next()

    (0 until nEpochs).foreach{i =>
      model.fit(mnistTrain)

      log.info("*** Completed epoch {} ***", i)

      // Feed forward to the embedding layer (layer 5) to get the 2d embedding to plot later
      val embedding: INDArray = model.feedForwardToLayer(5, testData.getFeatures).get(6)

      embeddingByEpoch.append((embedding, testData.getLabels))
      epochNum.append(i)
    }

    // Create a scatterplot: slider allows embeddings to be view at the end of each epoch
    PlotUtil.scatterPlot(embeddingByEpoch, epochNum,
      "MNIST Center Loss Embedding: l = " + lambda + ", a = " + alpha)
  }
}
