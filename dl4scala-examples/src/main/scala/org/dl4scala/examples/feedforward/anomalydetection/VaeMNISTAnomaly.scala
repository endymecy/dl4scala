package org.dl4scala.examples.feedforward.anomalydetection

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.layers.variational.{BernoulliReconstructionDistribution, VariationalAutoencoder}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.slf4j.{Logger, LoggerFactory}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * This example performs unsupervised anomaly detection on MNIST using a variational autoencoder,
  * trained with a Bernoulli reconstruction distribution.
  *
  * For details on the variational autoencoder, see:
  * - Kingma and Welling, 2013 - Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
  *
  * For the use of VAEs for anomaly detection using reconstruction probability see:
  * - An & Cho, 2015 - Variational Autoencoder based Anomaly Detection using Reconstruction Probability
  *   http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
  *
  *
  * Unsupervised training is performed on the entire data set at once in this example.
  * An alternative approach would be to train one model for each digit.
  *
  * After unsupervised training, examples are scored using the VAE layer (reconstruction probability).
  * Here, we are using the labels to get the examples with the highest and lowest reconstruction probabilities
  * for each digit for plotting. In a general unsupervised anomaly detection situation,
  * these labels would not be available, and hence highest/lowest probabilities for the entire data
  * set would be used instead.
  *
  * Created by endy on 2017/5/24.
  */
object VaeMNISTAnomaly extends App{
  val logger: Logger = LoggerFactory.getLogger(VaeMNISTAnomaly.getClass)

  implicit def ordered: Ordering[(Double, INDArray)] = new Ordering[(Double, INDArray)] {
    def compare(x: (Double, INDArray), y: (Double, INDArray)): Int = x._1 compareTo y._1
  }

  val minibatchSize = 128
  val rngSeed = 12345
  val nEpochs = 1
  // Total number of training epochs
  val reconstructionNumSamples = 16 // Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

  // MNIST data for training
  val trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed)

  // Neural net configuration
  Nd4j.getRandom.setSeed(rngSeed)

  val conf = new NeuralNetConfiguration.Builder()
    .seed(rngSeed)
    .learningRate(0.05)
    .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
    .weightInit(WeightInit.XAVIER)
    .regularization(true).l2(1e-4)
    .list()
    .layer(0, new VariationalAutoencoder.Builder()
      .activation(Activation.LEAKYRELU)
      .encoderLayerSizes(256, 256)                    // 2 encoder layers, each of size 256
      .decoderLayerSizes(256, 256)                    // 2 decoder layers, each of size 256
      .pzxActivationFunction(Activation.IDENTITY)     // p(z|data) activation function
      // Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
      .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
      .nIn(28 * 28)                                   // Input size: 28x28
      .nOut(32)                                       // Size of the latent variable space: p(z|x) - 32 values
      .build()).pretrain(true).backprop(false).build()

  val net = new MultiLayerNetwork(conf)
  net.init()
  net.setListeners(new ScoreIterationListener(100))

  for(i <- 0 until nEpochs){
    net.fit(trainIter)
    logger.info("Finished epoch " + (i + 1) + " of " + nEpochs)
  }

  // Perform anomaly detection on the test set, by calculating the reconstruction probability for each example
  // Then add pair (reconstruction probability, INDArray data) to lists and sort by score
  // This allows us to get best N and worst N digits for each digit type

  val testIter = new MnistDataSetIterator(minibatchSize, false, rngSeed)

  // Get the variational autoencoder layer:
  val vae = net.getLayer(0).asInstanceOf[org.deeplearning4j.nn.layers.variational.VariationalAutoencoder]

  var listsByDigit = new mutable.OpenHashMap[Int, ArrayBuffer[(Double, INDArray)]]
  (0 until 10).foreach(i => listsByDigit.put(i, new ArrayBuffer[(Double, INDArray)]()))

  // Iterate over the test data, calculating reconstruction probabilities
  while(testIter.hasNext) {
    val ds = testIter.next
    val features = ds.getFeatures
    val labels = Nd4j.argMax(ds.getLabels, 1) // Labels as integer indexes (from one hot), shape [minibatchSize, 1]
    val nRows = features.rows

    // Calculate the log probability for reconstructions as per An & Cho
    // Higher is better, lower is worse
    val reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples)

    for (j <- 0 until nRows) {
      val example = features.getRow(j)
      val label = labels.getDouble(j).asInstanceOf[Int]
      val score = reconstructionErrorEachExample.getDouble(j)
      listsByDigit(label).append((score, example))
    }
  }

  val sortedListByDigit = new mutable.OpenHashMap[Int, ArrayBuffer[(Double, INDArray)]]

  for(key <- listsByDigit.keys){
    val oneDigit = listsByDigit(key).sorted
    sortedListByDigit.put(key, oneDigit)
  }

  // After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
  val best = new ArrayBuffer[INDArray](50)
  val worst = new ArrayBuffer[INDArray](50)

  val bestReconstruction = new ArrayBuffer[INDArray](50)
  val worstReconstruction = new ArrayBuffer[INDArray](50)

  (0 until 10).foreach { i =>
    val list = sortedListByDigit(i)
    (0 until 5).foreach { j =>
      val w = list(j)._2
      val b = list(list.size - j - 1)._2

      val pzxMeanBest = vae.preOutput(b)
      val reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest)

      val pzxMeanWorst = vae.preOutput(w)
      val reconstructionWorst = vae.generateAtMeanGivenZ(pzxMeanWorst)

      best.append(b)
      bestReconstruction.append(reconstructionBest)
      worst.append(w)
      worstReconstruction.append(reconstructionWorst)
    }
  }

  // Visualize the best and worst digits//Visualize the best and worst digits
  val bestVisualizer = new MNISTVisualizer(2.0, best, "Best (Highest Rec. Prob)", 5)
  bestVisualizer.visualize()

  val bestReconstructions = new MNISTVisualizer(2.0, bestReconstruction, "Best - Reconstructions", 5)
  bestReconstructions.visualize()

  val worstVisualizer = new MNISTVisualizer(2.0, worst, "Worst (Lowest Rec. Prob)", 5)
  worstVisualizer.visualize()

  val worstReconstructions = new MNISTVisualizer(2.0, worstReconstruction, "Worst - Reconstructions", 5)
  worstReconstructions.visualize()
}
