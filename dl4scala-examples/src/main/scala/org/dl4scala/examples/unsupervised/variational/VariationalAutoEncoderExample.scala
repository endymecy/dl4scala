package org.dl4scala.examples.unsupervised.variational

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.dl4scala.examples.unsupervised.variational.plot.PlotUtil
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

/**
  * A simple example of training a variational autoencoder on MNIST.
  * This example intentionally has a small hidden state Z (2 values) for visualization on a 2-grid.
  *
  * After training, this example plots 2 things:
  * 1. The MNIST digit reconstructions vs. the latent space
  * 2. The latent space values for the MNIST test set, as training progresses (every N minibatches)
  *
  * Note that for both plots, there is a slider at the top - change this to see how the reconstructions and latent
  * space changes over time.
  *
  * Created by endy on 2017/6/25.
  */
object VariationalAutoEncoderExample {
  private val log = LoggerFactory.getLogger(VariationalAutoEncoderExample.getClass)

  def main(args: Array[String]): Unit = {
    val minibatchSize = 128
    val rngSeed = 12345
    val nEpochs = 20 // Total number of training epochs

    // Plotting configuration
    val plotEveryNMinibatches = 100 // Frequency with which to collect data for later plotting

    val plotMin = -5 // Minimum values for plotting (x and y dimensions)

    val plotMax = 5 // Maximum values for plotting (x and y dimensions)

    val plotNumSteps = 16 // Number of steps for reconstructions, between plotMin and plotMax

    // MNIST data for training
    val trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed)

    // Neural net configuration
    Nd4j.getRandom.setSeed(rngSeed)

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(1e-2)
      .updater(Updater.RMSPROP).rmsDecay(0.95)
      .weightInit(WeightInit.XAVIER)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new VariationalAutoencoder.Builder()
        .activation(Activation.LEAKYRELU)
        .encoderLayerSizes(256, 256)        //2 encoder layers, each of size 256
        .decoderLayerSizes(256, 256)        //2 decoder layers, each of size 256
        .pzxActivationFunction(Activation.IDENTITY)  //p(z|data) activation function
        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction))     //Bernoulli distribution for p(data|z) (binary or 0 to 1 data only)
        .nIn(28 * 28)                       //Input size: 28x28
        .nOut(2)                            //Size of the latent variable space: p(z|x). 2 dimensions here for plotting, use more in general
        .build())
      .pretrain(true).backprop(false).build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    //Get the variational autoencoder layer
    val vae = net.getLayer(0).asInstanceOf[(org.deeplearning4j.nn.layers.variational.VariationalAutoencoder)]

    //Test data for plotting
    val testdata = new MnistDataSetIterator(10000, false, rngSeed).next
    val testFeatures = testdata.getFeatures
    val testLabels = testdata.getLabels
    val latentSpaceGrid = getLatentSpaceGrid(plotMin, plotMax, plotNumSteps) //X/Y grid values, between plotMin and plotMax

    //Lists to store data for later plotting
    val latentSpaceVsEpoch = new ArrayBuffer[INDArray](nEpochs + 1)
    var latentSpaceValues = vae.activate(testFeatures, false);                     //Collect and record the latent space values before training starts
    latentSpaceVsEpoch.append(latentSpaceValues)
    val digitsGrid = new ArrayBuffer[INDArray]()

    var iterationCount = 0
    (0 until nEpochs).foreach{i =>
      log.info("Starting epoch {} of {}", i + 1, nEpochs)
      while (trainIter.hasNext) {
        val ds: DataSet = trainIter.next
        net.fit(ds)

        // Every N=100 minibatches:
        // (a) collect the test set latent space values for later plotting
        // (b) collect the reconstructions at each point in the grid
        if (iterationCount % plotEveryNMinibatches == 0) {
          latentSpaceValues = vae.activate(testFeatures, false)
          latentSpaceVsEpoch.append(latentSpaceValues)
          val out = vae.generateAtMeanGivenZ(latentSpaceGrid)
          digitsGrid.append(out)
          iterationCount += 1
        }
      }
      trainIter.reset()
    }

    //Plot MNIST test set - latent space vs. iteration (every 100 minibatches by default)

    PlotUtil.plotData(latentSpaceVsEpoch, testLabels, plotMin, plotMax, plotEveryNMinibatches)

    //Plot reconstructions - latent space vs. grid
    val imageScale = 2.0
    //Increase/decrease this to zoom in on the digits
    val v = new PlotUtil.MNISTLatentSpaceVisualizer(imageScale, digitsGrid, plotEveryNMinibatches)

    v.visualize()
  }

  //This simply returns a 2d grid: (x,y) for x=plotMin to plotMax, and y=plotMin to plotMax
  private def getLatentSpaceGrid(plotMin: Double, plotMax: Double, plotSteps: Int): INDArray = {
    val data = Nd4j.create(plotSteps * plotSteps, 2)
    val linspaceRow = Nd4j.linspace(plotMin, plotMax, plotSteps)

    (0 until plotSteps).foreach{i =>
      import org.nd4j.linalg.indexing.NDArrayIndex
      data.get(NDArrayIndex.interval(i * plotSteps, (i + 1) * plotSteps), NDArrayIndex.point(0)).assign(linspaceRow)
      val yStart = plotSteps - i - 1
      data.get(NDArrayIndex.interval(yStart * plotSteps, (yStart + 1) * plotSteps), NDArrayIndex.point(1)).assign(linspaceRow.getDouble(i))
    }

    data
  }
}
