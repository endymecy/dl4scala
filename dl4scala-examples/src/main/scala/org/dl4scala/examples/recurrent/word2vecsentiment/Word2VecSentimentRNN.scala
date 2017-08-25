package org.dl4scala.examples.recurrent.word2vecsentiment

import java.io.File
import java.net.URL

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration, Updater, WorkspaceMode}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.dl4scala.examples.utilities.DataUtilities
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

/**
  * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
  * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
  * (using the Word2Vec model) and fed into a recurrent neural network.
  * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
  * This data set contains 25,000 training reviews + 25,000 testing reviews
  *
  * Process:
  * 1. Automatic on first run of example: Download data (movie reviews) + extract
  * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
  * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
  * 4. Train network
  *
  * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
  * additional tuning.
  *
  * NOTE / INSTRUCTIONS:
  * You will have to download the Google News word vector model manually. ~1.5GB
  * The Google News vector model available here: https://code.google.com/p/word2vec/
  * Download the GoogleNews-vectors-negative300.bin.gz file
  * Then: set the WORD_VECTORS_PATH field to point to this location.
  * Created by endy on 2017/5/26.
  */
object Word2VecSentimentRNN extends App{
  val logger: Logger = LoggerFactory.getLogger(Word2VecSentimentRNN.getClass)

  // Data URL for downloading
  val DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  // Location to save and extract the training/testing data
  val DATA_PATH = new ClassPathResource("/w2vSentiment").getFile.getPath
  // Location (local file system) for the Google News vectors. Set this manually.
  val WORD_VECTORS_PATH = new ClassPathResource("/w2vSentiment/GoogleNews-vectors-negative300.bin.gz").getFile().getPath

  val batchSize = 64    //Number of examples in each minibatch
  val vectorSize = 300   //Size of the word vectors. 300 in the Google News model
  val nEpochs = 1      //Number of epochs (full passes of training data) to train on
  val truncateReviewsToLength = 256  //Truncate reviews with length (# words) greater than this

  Nd4j.getMemoryManager.setAutoGcWindow(10000)

  def downloadData(): Unit = {
    // Create directory if required
    val directory = new File(DATA_PATH)
    if (!directory.exists) directory.mkdir

    // Download file:
    val archizePath = DATA_PATH + "/aclImdb_v1.tar.gz"
    val archiveFile = new File(archizePath)
    val extractedPath = DATA_PATH + "/aclImdb"
    val extractedFile = new File(extractedPath)

    if (!archiveFile.exists) {
      logger.info("Starting data download (80MB)...")
      FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
      logger.info("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath)
      // Extract tar.gz file to output directory
      DataUtilities.extractTarGz(archizePath, DATA_PATH)
    } else {
      // Assume if archive (.tar.gz) exists, then data has already been extracted
      logger.info("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath)
      if (!extractedFile.exists) DataUtilities.extractTarGz(archizePath, DATA_PATH)
      else logger.info("Data (extracted) already exists at " + extractedFile.getAbsolutePath)
    }
  }

  // Download and extract data
  downloadData()

  // Set up network configuration
  val conf = new NeuralNetConfiguration.Builder()
    .updater(Updater.ADAM)
    .regularization(true).l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0)
    .learningRate(2e-2)
    .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
      .activation(Activation.TANH).build())
    .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
    .pretrain(false).backprop(true)
    .build()

  val net = new MultiLayerNetwork(conf)
  net.init()
  net.setListeners(new ScoreIterationListener(1))

  // DataSetIterators for training and testing respectively
  val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
  val train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true)
  val test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false)

  logger.info("Starting training")

  (0 until nEpochs).foreach{i =>
    net.fit(train)
    train.reset()
    logger.info("Epoch " + i + " complete. Starting evaluation:")

    //Run evaluation. This is on 25k reviews, so can take some time
    val evaluation = new Evaluation()
    while (test.hasNext) {
      val t = test.next()
      val features = t.getFeatureMatrix
      val labels = t.getLabels
      val inMask = t.getFeaturesMaskArray
      val outMask = t.getLabelsMaskArray
      val predicted = net.output(features, false, inMask, outMask)
      evaluation.evalTimeSeries(labels, predicted, outMask)
    }
    test.reset()

    logger.info(evaluation.stats)
  }


  //After training: load a single example and generate predictions

  val firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"))
  val firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile)

  val features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength)
  val networkOutput = net.output(features)
  val timeSeriesLength = networkOutput.size(2)
  val probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(timeSeriesLength - 1))

  logger.info("\n\n-------------------------------")
  logger.info("First positive review: \n" + firstPositiveReview)
  logger.info("\n\nProbabilities at last time step:")
  logger.info("p(positive): " + probabilitiesAtLastWord.getDouble(0))
  logger.info("p(negative): " + probabilitiesAtLastWord.getDouble(1))

  logger.info("----- Example complete -----")
}
