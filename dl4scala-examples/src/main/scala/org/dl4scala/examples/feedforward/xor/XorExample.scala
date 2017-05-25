package org.dl4scala.examples.feedforward.xor

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by endy on 2017/5/25.
  */
object XorExample extends App{
  val logger: Logger = LoggerFactory.getLogger(XorExample.getClass)

  // list off input values, 4 training samples with data for 2
  // input-neurons each
  val input = Nd4j.zeros(4, 2)

  // correspondending list with expected output values, 4 training samples
  // with data for 2 output-neurons each
  val labels = Nd4j.zeros(4, 2)

  // create first dataset
  // when first input=0 and second input=0
  input.putScalar(Array[Int](0, 0), 0)
  input.putScalar(Array[Int](0, 1), 0)

  // then the first output fires for false, and the second is 0 (see class
  // comment)
  labels.putScalar(Array[Int](0, 0), 1)
  labels.putScalar(Array[Int](0, 1), 0)

  // when first input=1 and second input=0
  input.putScalar(Array[Int](1, 0), 1)
  input.putScalar(Array[Int](1, 1), 0)

  // then xor is true, therefore the second output neuron fires
  labels.putScalar(Array[Int](1, 0), 0)
  labels.putScalar(Array[Int](1, 1), 1)

  // same as above
  input.putScalar(Array[Int](2, 0), 0)
  input.putScalar(Array[Int](2, 1), 1)
  labels.putScalar(Array[Int](2, 0), 0)
  labels.putScalar(Array[Int](2, 1), 1)

  // when both inputs fire, xor is false again - the first output should
  // fire
  input.putScalar(Array[Int](3, 0), 1)
  input.putScalar(Array[Int](3, 1), 1)
  labels.putScalar(Array[Int](3, 0), 1)
  labels.putScalar(Array[Int](3, 1), 0)


  // create dataset object
  val ds: DataSet = new DataSet(input, labels)

  // Set up network configuration
  val builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder
  // how often should the training set be run, we need something above
  // 1000, or a higher learning-rate - found this values just by trial and
  // error
  builder.iterations(10000)
  // learning rate
  builder.learningRate(0.1)
  // fixed seed for the random generator, so any run of this program
  // brings the same results - may not work if you do something like
  // ds.shuffle()
  builder.seed(123)
  // not applicable, this network is to small - but for bigger networks it
  // can help that the network will not only recite the training data
  builder.useDropConnect(false)
  // a standard algorithm for moving on the error-plane, this one works
  // best for me, LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT can do the
  // job, too - it's an empirical value which one matches best to
  // your problem
  builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
  // init the bias with 0 - empirical value, too

  builder.biasInit(0)
  // from "http://deeplearning4j.org/architecture": The networks can
  // process the input more quickly and more accurately by ingesting
  // minibatches 5-10 elements at a time in parallel.
  // this example runs better without, because the dataset is smaller than
  // the mini batch size
  builder.miniBatch(false)

  // create a multilayer network with 2 layers (including the output
  // layer, excluding the input payer)
  val listBuilder = builder.list

  val hiddenLayerBuilder = new DenseLayer.Builder
  // two input connections - simultaneously defines the number of input
  // neurons, because it's the first non-input-layer
  hiddenLayerBuilder.nIn(2)
  // number of outgooing connections, nOut simultaneously defines the
  // number of neurons in this layer
  hiddenLayerBuilder.nOut(4)
  // put the output through the sigmoid function, to cap the output
  // valuebetween 0 and 1
  hiddenLayerBuilder.activation(Activation.SIGMOID)
  // random initialize weights with values between 0 and 1
  hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION)
  hiddenLayerBuilder.dist(new UniformDistribution(0, 1))

  // build and set as layer 0
  listBuilder.layer(0, hiddenLayerBuilder.build)

  import org.deeplearning4j.nn.conf.MultiLayerConfiguration

  // MCXENT or NEGATIVELOGLIKELIHOOD (both are mathematically equivalent) work ok for this example - this
  // function calculates the error-value (aka 'cost' or 'loss function value'), and quantifies the goodness
  // or badness of a prediction, in a differentiable way
  // For classification (with mutually exclusive classes, like here), use multiclass cross entropy, in conjunction
  // with softmax activation function
  val outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
  // must be the same amout as neurons in the layer before
  outputLayerBuilder.nIn(4)
  // two neurons in this layer
  outputLayerBuilder.nOut(2)
  outputLayerBuilder.activation(Activation.SOFTMAX)
  outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION)
  outputLayerBuilder.dist(new UniformDistribution(0, 1))
  listBuilder.layer(1, outputLayerBuilder.build)

  // no pretrain phase for this network
  listBuilder.pretrain(false)

  // seems to be mandatory
  // according to agibsonccc: You typically only use that with
  // pretrain(true) when you want to do pretrain/finetune without changing
  // the previous layers finetuned weights that's for autoencoders and
  // rbms
  listBuilder.backprop(true)

  // build and init the network, will check if everything is configured
  // correct
  val conf = listBuilder.build
  val net = new MultiLayerNetwork(conf)

  // add an listener which outputs the error every 100 parameter updates// add an listener which outputs the error every 100 parameter updates

  net.setListeners(new ScoreIterationListener(100))

  // C&P from GravesLSTMCharModellingExample
  // Print the number of parameters in the network (and for each layer)
  val layers: Array[Layer] = net.getLayers

  val totalNumParams = layers.indices.map{ i =>
    val nParams = layers(i).numParams
    logger.info("Number of parameters in layer " + i + ": " + nParams)
    nParams
  }.sum

  // here the actual learning takes place
  net.fit(ds)

  // create output for every training sample
  val output = net.output(ds.getFeatureMatrix)
  logger.info(output.toString)

  // let Evaluation prints stats how often the right output had the
  // highest value
  val eval = new Evaluation(2)
  eval.eval(ds.getLabels, output)
  logger.info(eval.stats)

}
