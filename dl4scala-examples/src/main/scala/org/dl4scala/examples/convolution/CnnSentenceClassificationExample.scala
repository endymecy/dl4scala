package org.dl4scala.examples.convolution

import java.io.File

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.{ConvolutionMode, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, GlobalPoolingLayer, OutputLayer, PoolingType}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.dl4scala.examples.recurrent.word2vecsentiment.Word2VecSentimentRNN
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.util.Random

/**
  * Created by endy on 2017/5/26.
  */
object CnnSentenceClassificationExample extends App{
  val logger: Logger = LoggerFactory.getLogger(CnnSentenceClassificationExample.getClass)

  // Data URL for downloading
  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  // Location to save and extract the training/testing data
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/")
  // Location (local file system) for the Google News vectors. Set this manually.
  val WORD_VECTORS_PATH = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin.gz"

  // Download and extract data
  Word2VecSentimentRNN.downloadData()

  // Basic configuration//Basic configuration
  val batchSize = 32
  val vectorSize = 300 // Size of the word vectors. 300 in the Google News model
  val nEpochs = 1 // Number of epochs (full passes of training data) to train on
  val truncateReviewsToLength = 256 // Truncate reviews with length (# words) greater than this
  val cnnLayerFeatureMaps = 100 // Number of feature maps / channels / depth for each CNN layer
  val globalPoolingType = PoolingType.MAX
  val rng = new Nothing(12345) // For shuffling repeatability

  // Set up the network configuration. Note that we have multiple convolution layers, each wih filter
  // widths of 3, 4 and 5 as per Kim (2014) paper.
  val config = new NeuralNetConfiguration.Builder()
    .weightInit(WeightInit.RELU)
    .activation(Activation.LEAKYRELU)
    .updater(Updater.ADAM)
    .convolutionMode(ConvolutionMode.Same)      // This is important so we can 'stack' the results later
    .regularization(true).l2(0.0001)
    .learningRate(0.01)
    .graphBuilder()
    .addInputs("input")
    .addLayer("cnn3", new ConvolutionLayer.Builder() // 3 gram
      .kernelSize(3, vectorSize)
      .stride(1, vectorSize)
      .nIn(1)
      .nOut(cnnLayerFeatureMaps)
      .build(), "input")
    .addLayer("cnn4", new ConvolutionLayer.Builder() // 4 gram
      .kernelSize(4,vectorSize)
      .stride(1,vectorSize)
      .nIn(1)
      .nOut(cnnLayerFeatureMaps)
      .build(), "input")
    .addLayer("cnn5", new ConvolutionLayer.Builder() // 5 gram
      .kernelSize(5,vectorSize)
      .stride(1,vectorSize)
      .nIn(1)
      .nOut(cnnLayerFeatureMaps)
      .build(), "input")
    .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
    .addLayer("globalPool", new GlobalPoolingLayer.Builder()
    .poolingType(globalPoolingType)
    .build(), "merge")
    .addLayer("out", new OutputLayer.Builder()
      .lossFunction(LossFunctions.LossFunction.MCXENT)
      .activation(Activation.SOFTMAX)
      .nIn(3*cnnLayerFeatureMaps)
      .nOut(2)    //2 classes: positive or negative
      .build(), "globalPool")
    .setOutputs("out")
    .build()

  val net = new ComputationGraph(config)
  net.init()

  logger.info("Number of parameters by layer:")

  for (l <- net.getLayers) {
    logger.info("\t" + l.conf.getLayer.getLayerName + "\t" + l.numParams)
  }

  // Load word vectors and get the DataSetIterators for training and testing
  logger.info("Loading word vectors and creating DataSetIterators")


  val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
  val trainIter = getDataSetIterator(isTraining = true, wordVectors, batchSize, truncateReviewsToLength, rng)
  val testIter = getDataSetIterator(isTraining = false, wordVectors, batchSize, truncateReviewsToLength, rng)

  logger.info("Starting training")

  (0 until nEpochs).foreach{i =>
    net.fit(trainIter)
    logger.info("Epoch " + i + " complete. Starting evaluation:")
    //Run evaluation. This is on 25k reviews, so can take some time
    val evaluation = net.evaluate(testIter)
    logger.info(evaluation.stats)
  }


  //After training: load a single sentence and generate a prediction

  val pathFirstNegativeFile = FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/0_2.txt")
  val contentsFirstNegative = FileUtils.readFileToString(new File(pathFirstNegativeFile))
  val featuresFirstNegative = testIter.asInstanceOf[CnnSentenceDataSetIterator].loadSingleSentence(contentsFirstNegative)

  val predictionsFirstNegative = net.outputSingle(featuresFirstNegative)
  val labels = testIter.getLabels

  for(i <- 0 until labels.size()){
    logger.info("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i))
  }


  private def getDataSetIterator(isTraining: Boolean, wordVectors: WordVectors, minibatchSize: Int,
                                 maxSentenceLength: Int, rng: Random): DataSetIterator = {
    val path = FilenameUtils.concat(DATA_PATH, if (isTraining) "aclImdb/train/" else "aclImdb/test/")
    val positiveBaseDir = FilenameUtils.concat(path, "pos")
    val negativeBaseDir = FilenameUtils.concat(path, "neg")

    val filePositive = new File(positiveBaseDir)
    val fileNegative = new File(negativeBaseDir)

    val reviewFilesMap = new mutable.OpenHashMap[String, java.util.List[File]]()

    reviewFilesMap.put("Positive", filePositive.listFiles.toList.asJava)
    reviewFilesMap.put("Negative", fileNegative.listFiles.toList.asJava)

    val sentenceProvider = new FileLabeledSentenceProvider(reviewFilesMap.asJava)

    new CnnSentenceDataSetIterator.Builder()
      .sentenceProvider(sentenceProvider)
      .wordVectors(wordVectors)
      .minibatchSize(minibatchSize)
      .maxSentenceLength(maxSentenceLength)
      .useNormalizedWordVectors(false)
      .build()
  }
}
