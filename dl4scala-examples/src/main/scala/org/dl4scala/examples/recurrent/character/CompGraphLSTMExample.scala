package org.dl4scala.examples.recurrent.character

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.dl4scala.examples.recurrent.character.GravesLSTMCharModellingExample.{getShakespeareIterator, sampleFromDistribution}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.util.Random

/**
  * This example is almost identical to the GravesLSTMCharModellingExample, except that it utilizes the ComputationGraph
  * architecture instead of MultiLayerNetwork architecture. See the javadoc in that example for details.
  * For more details on the ComputationGraph architecture, see http://deeplearning4j.org/compgraph
  *
  * In addition to the use of the ComputationGraph a, this version has skip connections between the first and output layers,
  * in order to show how this configuration is done. In practice, this means we have the following types of connections:
  * (a) first layer -> second layer connections
  * (b) first layer -> output layer connections
  * (c) second layer -> output layer connections
  *
  * Created by endy on 2017/6/20.
  */
object CompGraphLSTMExample {
  def main(args: Array[String]): Unit = {
    val lstmLayerSize = 200					//Number of units in each GravesLSTM layer
    val miniBatchSize = 32					//Size of mini batch to use when  training
    val exampleLength = 1000				//Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 50                      //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1							//Total number of training epochs
    val generateSamplesEveryNMinibatches = 10 //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 4					//Number of samples to generate after each training epoch
    val nCharactersToSample = 300				//Length of each sample to generate
    val generationInitialization: String = null	//Optional character initialization; a random character is used if null
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    val rng = new Random(12345)

    // Get a DataSetIterator that handles vectorization of text into something we can use to train
    // our GravesLSTM network.
    val iter = getShakespeareIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes()

    //Set up network configuration:
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .learningRate(0.1)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input") //Give the input a name. For a ComputationGraph with multiple inputs, this also defines the input array orders
      //First layer: name "first", with inputs from the input called "input"
      .addLayer("first", new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
      .updater(Updater.RMSPROP).activation(Activation.TANH).build(),"input")
      //Second layer, name "second", with inputs from the layer called "first"
      .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
      .updater(Updater.RMSPROP)
      .activation(Activation.TANH).build(),"first")
      //Output layer, name "outputlayer" with inputs from the two layers called "first" and "second"
      .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
      .activation(Activation.SOFTMAX).updater(Updater.RMSPROP)
      .nIn(2*lstmLayerSize).nOut(nOut).build(),"first","second")
      .setOutputs("outputLayer")  //List the output. For a ComputationGraph with multiple outputs, this also defines the input array orders
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build()

    val net = new ComputationGraph(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    // Print the  number of parameters in the network (and for each layer)
    val layers = net.getLayers
    var totalNumParams = 0

    (0 until layers.length).foreach{i =>
      val nParams = layers(i).numParams
      System.out.println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    System.out.println("Total number of network parameters: " + totalNumParams)

    var miniBatchNumber = 0
    (0 until numEpochs).foreach{i =>
      while(iter.hasNext()){
        val ds = iter.next()
        net.fit(ds)
        miniBatchNumber += 1
        if(miniBatchNumber % generateSamplesEveryNMinibatches == 0){
          System.out.println("--------------------")
          System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
          System.out.println("Sampling characters from network given initialization \"" + (if (generationInitialization == null) ""
          else generationInitialization) + "\"")
          val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
          samples.indices.foreach{j =>
            System.out.println("----- Sample " + j + " -----")
            System.out.println(samples(j))
            System.out.println()
          }
        }
      }
      iter.reset()
    }
    System.out.println("\n\nExample complete")
  }


  import org.deeplearning4j.nn.graph.ComputationGraph

  /**
    * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
    * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
    * Note that the initalization is used for all samples
    *
    * @param initialization1     String, may be null. If null, select a random character as initialization for all samples
    * @param charactersToSample Number of characters to sample from network (excluding initialization)
    * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
    * @param iter               CharacterIterator. Used for going from indexes back to characters
    */
  private def sampleCharactersFromNetwork(initialization1: String, net: ComputationGraph, iter: CharacterIterator,
                                          rng: Random, charactersToSample: Int, numSamples: Int) = {
    // Set up initialization. If no initialization: use a random character
    val initialization = if (initialization1 == null) String.valueOf(iter.getRandomCharacter) else initialization1

    val initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length)
    val init = initialization.toCharArray
    init.indices.foreach{i =>
      val idx = iter.convertCharacterToIndex(init(i))
      (0 until numSamples).foreach{j =>
        initializationInput.putScalar(Array[Int](j, idx, i), 1.0f)
      }
    }

    val sb = new Array[StringBuilder](numSamples)

    (0 until numSamples).foreach(i => sb(i) = new StringBuilder(initialization))

    // Sample from network (and feed samples back into input) one character at a time (for all samples)
    // Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)(0)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    (0 until charactersToSample).foreach{i =>
      // Set up next input (single time step) by sampling from previous output
      val nextInput = Nd4j.zeros(numSamples,iter.inputColumns())

      (0 until numSamples).foreach{s =>
        val outputProbDistribution = new Array[Double](iter.totalOutcomes())
        outputProbDistribution.indices.foreach(j => outputProbDistribution(j) = output.getDouble(s, j))
        val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
        nextInput.putScalar(Array[Int](s, sampledCharacterIdx), 1.0f) //Prepare next time step input
        sb(s).append(iter.convertIndexToCharacter(sampledCharacterIdx))
      }
      output = net.rnnTimeStep(nextInput)(0)
    }
    val out = new Array[String](numSamples)
    (0 until numSamples).foreach(i => out(i) = sb(i).toString)
    out
  }
}
