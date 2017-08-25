package org.dl4scala.examples.feedforward.classification.detectgender

import java.io.File

import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by endy on 2017/5/13.
  */
object PredictGenderTrain extends App{
  private val log: Logger = LoggerFactory.getLogger(PredictGenderTrain.getClass)

  var filePath = new ClassPathResource("/PredictGender/Data/").getFile.getPath
  val seed = 123456
  val learningRate = 0.005

  val batchSize = 100
  val nEpochs = 1
  var numInputs = 0
  var numOutputs = 0
  var numHiddenNodes = 0


  val arrayBuffer = new ArrayBuffer[String]()
  arrayBuffer.append("M")
  arrayBuffer.append("F")
  val rr = new GenderRecordReader(arrayBuffer)
  val rr1 = new GenderRecordReader(arrayBuffer)

  val st = System.currentTimeMillis
  log.info("Preprocessing start time : " + st)
  rr.initialize(new FileSplit(new File(filePath)))
  rr1.initialize(new FileSplit(new File(filePath)))

  val et = System.currentTimeMillis
  log.info("Preprocessing end time : " + et)
  log.info("time taken to process data : " + (et - st) + " ms")

  numInputs = rr.maxLengthName * 5 // multiplied by 5 as for each letter we use five binary digits like 00000
  numOutputs = 2
  numHiddenNodes = 2 * numInputs + numOutputs

  val trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2)
  val testIter = new RecordReaderDataSetIterator(rr1, batchSize, numInputs, 2)

  val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .biasInit(1)
      .iterations(1)
      .regularization(true).l2(1e-4)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.NESTEROVS)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .nIn(numHiddenNodes).nOut(numOutputs).build())
      .pretrain(false).backprop(true)
      .build()

  val model = new MultiLayerNetwork(conf)
  model.init()
  model.setListeners(new ScoreIterationListener(10)) // Print score every 10 parameter updates

  for(_ <- 0 until nEpochs){
      while (trainIter.hasNext) model.fit(trainIter.next())
      trainIter.reset()
  }

  ModelSerializer.writeModel(model, this.filePath + "/PredictGender1.net", true)

  log.info("Evaluate model....")
  val eval = new Evaluation(numOutputs)
  while (testIter.hasNext){
    val t = testIter.next
    val predicted = model.output(t.getFeatureMatrix, false)
    eval.eval(t.getLabels, predicted)
  }
  log.info(eval.stats())
}
