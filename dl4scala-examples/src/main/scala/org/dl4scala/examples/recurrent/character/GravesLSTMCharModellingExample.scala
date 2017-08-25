package org.dl4scala.examples.recurrent.character

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

/**
  * Example: Train a LSTM RNN to generates text, one character at a time.
	* This example is somewhat inspired by Andrej Karpathy's blog post,
	* "The Unreasonable Effectiveness of Recurrent Neural Networks"
	* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
	* This example is set up to train on the Complete Works of William Shakespeare, downloaded
	* from Project Gutenberg. Training on other text sources should be relatively easy to implement.
  *   For more details on RNNs in DL4J, see the following:
  *   http://deeplearning4j.org/usingrnns
  *   http://deeplearning4j.org/lstm
  *   http://deeplearning4j.org/recurrentnetwork
  *
  * Created by endy on 2017/6/20.
  */
object GravesLSTMCharModellingExample {
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
      // Set up network configuration:
      val conf = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
        .learningRate(0.1)
        .seed(12345)
        .regularization(true)
        .l2(0.001)
        .weightInit(WeightInit.XAVIER)
        .updater(Updater.RMSPROP)
        .list()
        .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
          .activation(Activation.TANH).build())
        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .activation(Activation.TANH).build())
        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
          .nIn(lstmLayerSize).nOut(nOut).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(false).backprop(true)
        .build()

      val net = new MultiLayerNetwork(conf)
      net.init()
      net.setListeners(new ScoreIterationListener(1))

      // Print the  number of parameters in the network (and for each layer)
      val layers = net.getLayers
      var totalNumParams = 0

      (0 until layers.length).foreach{i =>
        val nParams = layers(i).numParams
        println("Number of parameters in layer " + i + ": " + nParams)
        totalNumParams += nParams
      }
      println("Total number of network parameters: " + totalNumParams)

      var miniBatchNumber = 0
      (0 until numEpochs).foreach{i =>
        while(iter.hasNext()){
          val ds = iter.next()
          net.fit(ds)
          miniBatchNumber += 1
          if(miniBatchNumber % generateSamplesEveryNMinibatches == 0){
            println("--------------------")
            println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
            println("Sampling characters from network given initialization \"" + (if (generationInitialization == null) ""
            else generationInitialization) + "\"")
            val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
            samples.indices.foreach{j =>
              println("----- Sample " + j + " -----")
              println(samples(j))
              println()
            }
          }
        }
        iter.reset()
      }
      println("\n\nExample complete")
    }

  /**
    * Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
    * DataSetIterator that does vectorization based on the text.
    * @param miniBatchSize Number of text segments in each training mini-batch
    * @param sequenceLength Number of characters in each text segment.
    */
  def  getShakespeareIterator(miniBatchSize: Int, sequenceLength: Int): CharacterIterator = {
      //The Complete Works of William Shakespeare
      val fileLocation = new ClassPathResource("character/Shakespeare.txt").getFile.getPath
      println(fileLocation)

      val validCharacters = CharacterIterator.getMinimalCharacterSet //Which characters are allowed? Others will be removed

      new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
        miniBatchSize, sequenceLength, validCharacters, new Random(12345))
  }

  private def sampleCharactersFromNetwork(initialization1: String, net: MultiLayerNetwork, iter: CharacterIterator,
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
    var output = net.rnnTimeStep(initializationInput)
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
      output = net.rnnTimeStep(nextInput)
    }
    val out = new Array[String](numSamples)
    (0 until numSamples).foreach(i => out(i) = sb(i).toString)
    out
  }

  def sampleFromDistribution(distribution: Array[Double], rng: Random): Int = {
    var d = 0.0
    var sum = 0.0

    (0 until 10).foreach{t =>
      d = rng.nextDouble
      sum = 0.0
      distribution.indices.foreach{i =>
        sum += distribution(i)
        if( d <= sum ) return i
      }
    }
    throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum)
  }
}
