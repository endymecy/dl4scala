package org.dl4scala.examples.recurrent.regression

import javax.swing.{JFrame, WindowConstants}

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

/**
  * Created by endy on 2017/5/27.
  */
object SingleTimestepRegressionExample extends App{
  private val logger = LoggerFactory.getLogger(SingleTimestepRegressionExample.getClass)

  val baseDir = new ClassPathResource("/rnnRegression").getFile
  val miniBatchSize = 32
  val nEpochs = 300

  // Load the training data
  val trainReader = new CSVSequenceRecordReader(0, ";")
  trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath + "/passengers_train_%d.csv", 0, 0))
  // For regression, numPossibleLabels is not used. Setting it to -1 here
  val trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true)

  val testReader = new CSVSequenceRecordReader(0, ";")
  testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath + "/passengers_test_%d.csv", 0, 0))
  val testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true)

  // Create data set from iterator here since we only have a single data set
  val trainData = trainIter.next()
  val testData = testIter.next()

  // Normalize data, including labels (fitLabel=true)
  val normalizer = new NormalizerMinMaxScaler(0, 1)
  normalizer.fitLabel(true)
  normalizer.fit(trainData) // Collect training data statistics

  normalizer.transform(trainData)
  normalizer.transform(testData)

  // Configure the network
  val conf = new NeuralNetConfiguration.Builder()
    .seed(140)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .iterations(1)
    .weightInit(WeightInit.XAVIER)
    .updater(Updater.NESTEROVS)
    .learningRate(0.0015)
    .list
    .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build)
    .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(10).nOut(1).build)
    .build

  val net = new MultiLayerNetwork(conf)
  net.init()

  net.setListeners(new ScoreIterationListener(20))

  (0 until nEpochs).foreach{epoch =>
    net.fit(trainData)
    logger.info("Epoch " + epoch + " complete. Time series evaluation:")

    // Run regression evaluation on our single column input
    val evaluation = new RegressionEvaluation(1)
    val features = testData.getFeatureMatrix

    val labels = testData.getLabels
    val predicted = net.output(features, false)

    evaluation.evalTimeSeries(labels, predicted)

    // Just do sout here since the logger will shift the shift the columns of the stats
    logger.info(evaluation.stats)
  }


  //Init rrnTimeStemp with train data and predict test data

  net.rnnTimeStep(trainData.getFeatureMatrix)
  val predicted = net.rnnTimeStep(testData.getFeatureMatrix)

  // Revert data back to original values for plotting
  normalizer.revert(trainData)
  normalizer.revert(testData)
  normalizer.revertLabels(predicted)

  val trainFeatures = trainData.getFeatures
  val testFeatures = testData.getFeatures
  //Create plot with out data
  val c = new XYSeriesCollection
  createSeries(c, trainFeatures, 0, "Train data")
  createSeries(c, testFeatures, 99, "Actual test data")
  createSeries(c, predicted, 100, "Predicted test data")

  plotDataset(c)

  logger.info("----- Example Complete -----")


  private def createSeries(seriesCollection: XYSeriesCollection, data: INDArray, offset: Int, name: String) = {
    val nRows = data.shape()(2)
    val series = new XYSeries(name)
    for (i <- 0 until nRows) {
      series.add(i + offset, data.getDouble(i))
    }
    seriesCollection.addSeries(series)
  }

  /**
    * Generate an xy plot of the datasets provided.
    */
  private def plotDataset(c: XYSeriesCollection) = {
    val title = "Regression example"
    val xAxisLabel = "Timestep"
    val yAxisLabel = "Number of passengers"
    val orientation = PlotOrientation.VERTICAL
    val legend = true
    val tooltips = false
    val urls = false
    val chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls)

    // get a reference to the plot for further customisation...
    val plot = chart.getXYPlot
    // Auto zoom to fit time series in initial window
    val rangeAxis = plot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setAutoRange(true)

    val panel = new ChartPanel(chart)

    val f: JFrame = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Training Data")

    RefineryUtilities.centerFrameOnScreen(f)
    f.setVisible(true)
  }
}
