package org.dl4scala.examples.recurrent.processnews

import java.io.File

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

/**
  * This program trains a RNN to predict category of a news headlines. It uses word vector generated from PrepareWordVector.java.
  * - Labeled News are stored in \dl4scala-examples\src\main\resources\NewsData\LabelledNews folder in train and test folders.
  * - categories.txt file in \dl4scala-examples\src\main\resources\NewsData\LabelledNews folder contains category code and description.
  * - This categories are used along with actual news for training.
  * - news word vector is contained  in \dl4scala-examples\src\main\resources\NewsData\NewsWordVector.txt file.
  * - Trained model is stored in \dl4scala-examples\src\main\resources\NewsData\NewsModel.net file
  * - News Data contains only 3 categories currently.
  * - Data set structure is as given below
  * - categories.txt - this file contains various categories in category id,category description format. Sample categories are as below
  * 0,crime
  * 1,politics
  * 2,bollywood
  * 3,Business&Development
  * - For each category id above, there is a file containig actual news headlines, e.g.
  * 0.txt - contains news for crime headlines
  * 1.txt - contains news for politics headlines
  * 2.txt - contains news for bollywood
  * 3.txt - contains news for Business&Development
  * - You can add any new category by adding one line in categories.txt and respective news file in folder mentioned above.
  * - Below are training results with the news data given with this example.
  *  Scores
  * Accuracy:        0.9343
  * Precision:       0.9249
  * Recall:          0.9327
  * F1 Score:        0.9288
  *
  * <p>
  * Note :
  * - This code is a modification of original example named Word2VecSentimentRNN.java
  * - Results may vary with the data you use to train this network
  * <p>
  * <b>KIT Solutions Pvt. Ltd. (www.kitsol.com)</b>
  *
  * Created by endy on 2017/6/4.
  */
object TrainNews {
  private val logger: Logger = LoggerFactory.getLogger(TrainNews.getClass)

  private val userDirectory = new ClassPathResource("NewsData").getFile.getAbsolutePath + File.separator
  private val DATA_PATH = userDirectory + "LabelledNews"
  private val WORD_VECTORS_PATH = userDirectory + "NewsWordVector.txt"

  private val batchSize = 50    //Number of examples in each minibatch
  private val nEpochs = 1000       //Number of epochs (full passes of training data) to train on
  private val truncateReviewsToLength = 300  //Truncate reviews with length (# words) greater than this

  // DataSetIterators for training and testing respectively
  // Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
  val wordVectors: WordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))

  private var tokenizerFactory = new DefaultTokenizerFactory
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

  private val iTrain = new NewsIterator.Builder()
    .dataDirectory(DATA_PATH)
    .wordVectors(wordVectors)
    .batchSize(batchSize)
    .truncateLength(truncateReviewsToLength)
    .tokenizerFactory(tokenizerFactory)
    .train(true)
    .build()

  private val iTest = new NewsIterator.Builder()
    .dataDirectory(DATA_PATH)
    .wordVectors(wordVectors)
    .batchSize(batchSize)
    .tokenizerFactory(tokenizerFactory)
    .truncateLength(truncateReviewsToLength)
    .train(false)
    .build()

  private val inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length // 100 in our case
  private val outputs = iTrain.getLabels().size()

  tokenizerFactory = new DefaultTokenizerFactory
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

  //Set up network configuration
  val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .updater(Updater.RMSPROP).regularization(true).l2(1e-5)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0).learningRate(0.0018)
    .list
    .layer(0, new GravesLSTM.Builder().nIn(inputNeurons).nOut(200).activation(Activation.SOFTSIGN).build)
    .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(outputs).build)
    .pretrain(false).backprop(true)
    .build

  val net = new MultiLayerNetwork(conf)
  net.init()
  net.setListeners(new ScoreIterationListener(1))

  logger.info("Starting training")

  (0 until nEpochs).foreach{epoch =>
    net.fit(iTrain)
    iTrain.reset()
    logger.info("Epoch " + epoch + " complete. Starting evaluation:")
    //Run evaluation. This is on 25k reviews, so can take some time
    val evaluation = new Evaluation()

    while (iTest.hasNext()) {
      val t = iTest.next()
      val features = t.getFeatureMatrix
      val lables = t.getLabels
      val inMask = t.getFeaturesMaskArray
      val outMask = t.getLabelsMaskArray
      val predicted = net.output(features, false)
      evaluation.evalTimeSeries(lables, predicted, outMask)
    }

    iTest.reset()
    logger.info(evaluation.stats)
  }

  ModelSerializer.writeModel(net, userDirectory + "NewsModel.net", true)
  logger.info("----- Example complete -----")
}
