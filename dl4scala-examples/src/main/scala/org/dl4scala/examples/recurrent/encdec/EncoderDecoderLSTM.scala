package org.dl4scala.examples.recurrent.encdec

import java.io.{ByteArrayInputStream, File, FileNotFoundException, IOException}
import java.nio.charset.StandardCharsets
import java.util
import java.util.Map.Entry
import java.util.{Comparator, Date, Random, Scanner}
import java.util.concurrent.TimeUnit

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.rnn.{DuplicateToTimeSeriesVertex, LastTimeStepVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{EmbeddingLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, GradientNormalization, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._
import scala.collection.JavaConverters._

/**
  *  This is a seq2seq encoder-decoder LSTM model made according to the Google's paper: [1] The model tries to predict the next dialog
  * line using the provided one. It learns on the Cornell Movie Dialogs corpus. Unlike simple char RNNs this model is more sophisticated
  * and theoretically, given enough time and data, can deduce facts from raw text. Your mileage may vary. This particular network
  * architecture is based on AdditionRNN but changed to be used with a huge amount of possible tokens (10-40k) instead of just digits.
  *
  * Use the get_data.sh script to download, extract and optimize the train data. It's been only tested on Linux, it could work on OS X or
  * even on Windows 10 in the Ubuntu shell.
  *
  * Special tokens used:
  *
  * <unk> - replaces any word or other token that's not in the dictionary (too rare to be included or completely unknown)
  *
  * <eos> - end of sentence, used only in the output to stop the processing; the model input and output length is limited by the ROW_SIZE
  * constant.
  *
  * <go> - used only in the decoder input as the first token before the model produced anything
  *
  * The architecture is like this: Input => Embedding Layer => Encoder => Decoder => Output (softmax)
  *
  * The encoder layer produces a so called "thought vector" that contains a neurally-compressed representation of the input. Depending on
  * that vector the model produces different sentences even if they start with the same token. There's one more input, connected directly
  * to the decoder layer, it's used to provide the previous token of the output. For the very first output token we send a special <go>
  * token there, on the next iteration we use the token that the model produced the last time. On the training stage everything is
  * simple, we apriori know the desired output so the decoder input would be the same token set prepended with the <go> token and without
  * the last <eos> token. Example:
  *
  * Input: "how" "do" "you" "do" "?"
  *
  * Output: "I'm" "fine" "," "thanks" "!" "<eos>"
  *
  * Decoder: "<go>" "I'm" "fine" "," "thanks" "!"
  *
  * Actually, the input is reversed as per [2], the most important words are usually in the beginning of the phrase and they would get
  * more weight if supplied last (the model "forgets" tokens that were supplied "long ago", i.e. they have lesser weight than the recent
  * ones). The output and decoder input sequence lengths are always equal. The input and output could be of any length (less than
  * ROW_SIZE) so for purpose of batching we mask the unused part of the row. The encoder and decoder layers work sequentially. First the
  * encoder creates the thought vector, that is the last activations of the layer. Those activations are then duplicated for as many time
  * steps as there are elements in the output so that every output element can have its own copy of the thought vector. Then the decoder
  * starts working. It receives two inputs, the thought vector made by the encoder and the token that it _should have produced_ (but
  * usually it outputs something else so we have our loss metric and can compute gradients for the backward pass) on the previous step
  * (or <go> for the very first step). These two vectors are simply concatenated by the merge vertex. The decoder's output goes to the
  * softmax layer and that's it.
  *
  * The test phase is much more tricky. We don't know the decoder input because we don't know the output yet (unlike in the train phase),
  * it could be anything. So we can't use methods like outputSingle() and have to do some manual work. Actually, we can but it would
  * require full restarts of the entire process, it's super slow and ineffective.
  *
  * First, we do a single feed forward pass for the input with a single decoder element, <go>. We don't need the actual activations
  * except the "thought vector". It resides in the second merge vertex input (named "dup"). So we get it and store for the entire
  * response generation time. Then we put the decoder input (<go> for the first iteration) and the thought vector to the merge vertex
  * inputs and feed it forward. The result goes to the decoder layer, now with rnnTimeStep() method so that the internal layer state is
  * updated for the next iteration. The result is fed to the output softmax layer and then we sample it randomly (not with argMax(), it
  * tends to give a lot of same tokens in a row). The resulting token is looked up in the dictionary, printed to the stdout and then it
  * goes to the next iteration as the decoder input and so on until we get <eos>.
  *
  * To continue the training process from a specific batch number, enter it when prompted; batch numbers are printed after each processed
  * macrobatch. If you've changed the minibatch size after the last launch, recalculate the number accordingly, i.e. if you doubled the
  * minibatch size, specify half of the value and so on.
  *
  * [1] https://arxiv.org/abs/1506.05869 A Neural Conversational Model
  *
  * [2] https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf Sequence to Sequence Learning with
  * Neural Networks
  *
  * Created by endy on 2017/6/3.
  */
class EncoderDecoderLSTM {
  private val logger: Logger = LoggerFactory.getLogger(classOf[EncoderDecoderLSTM])

  private val dict = new mutable.OpenHashMap[String, Double]()
  private val revDict = new mutable.OpenHashMap[Double, String]()
  private val CHARS = "-\\/_&" + CorpusProcessor.SPECIALS
  private val corpus = new ArrayBuffer[ArrayBuffer[Double]]()
  private val HIDDEN_LAYER_WIDTH = 512 // this is purely empirical, affects performance and VRAM requirement

  private val EMBEDDING_WIDTH = 128 // one-hot vectors will be embedded to more dense vectors with this width

  private val CORPUS_FILENAME = "movie_lines.txt" // filename of data corpus to learn

  private val MODEL_FILENAME = "rnn_train.zip" // filename of the model

  private val BACKUP_MODEL_FILENAME = "rnn_train.bak.zip" // filename of the previous version of the model (backup)

  private val MINIBATCH_SIZE = 32
  private val rnd = new Random(new Date().getTime)
  private val SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5) // save the model with this period

  private val TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1) // test the model with this period

  private val MAX_DICT = 20000 // this number of most frequent words will be used, unknown words (that are not in the

  // dictionary) are replaced with <unk> token
  private val TBPTT_SIZE = 25
  private val LEARNING_RATE = 1e-1
  private val RMS_DECAY = 0.95
  private val ROW_SIZE = 40 // maximum line length in tokens

  private val GC_WINDOW = 2000 // delay between garbage collections, try to reduce if you run out of VRAM or increase for

  // better performance
  private val MACROBATCH_SIZE = 20 // see CorpusIterator

  private var net: ComputationGraph = _

  @throws(classOf[IOException])
  private def run(args: Array[String]) = {
    Nd4j.getMemoryManager.setAutoGcWindow(GC_WINDOW)
    createDictionary()
    val networkFile = new File(toTempPath(MODEL_FILENAME))
    var offset = 0
    if (networkFile.exists()) {
      logger.info("Loading the existing network...")
      net = ModelSerializer.restoreComputationGraph(networkFile)
      logger.info("Enter d to start dialog or a number to continue training from that minibatch: ")

      val scanner = new Scanner(System.in)
      Stream.continually(scanner.nextLine()).foreach{line =>
        if (line.toLowerCase.equals("d")) startDialog(scanner)
        else {
          offset = Integer.valueOf(line)
          test()
        }
      }
    } else {
      logger.info("Creating a new network...")
      createComputationGraph()
    }

    logger.info("Number of parameters: " + net.numParams)
    net.setListeners(new ScoreIterationListener(1))
    train(networkFile, offset)
  }

  @throws(classOf[IOException])
  @throws(classOf[FileNotFoundException])
  private def createDictionary() = {
    var idx = 3.0
    dict.put("<unk>", 0.0)
    revDict.put(0.0, "<unk>")
    dict.put("<eos>", 1.0)
    revDict.put(1.0, "<eos>")
    dict.put("<go>", 2.0)
    revDict.put(2.0, "<go>")

    for (c <- CHARS.toCharArray) {
      if (!dict.contains(c.toString)) {
        dict.put(String.valueOf(c), idx)
        revDict.put(idx, String.valueOf(c))
        idx += 1
      }
    }
    logger.info("Building the dictionary...")

    var corpusProcessor: CorpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, true)
    corpusProcessor.start()

    val freqs = corpusProcessor.getFreq
    val dictSet = new mutable.TreeSet[String]() // the tokens order is preserved for TreeSet
    val freqMap = new util.TreeMap[Double, mutable.TreeSet[String]](new Comparator[Double]() {
      def compare(o1: Double, o2: Double): Int = (o2 - o1).toInt
    })

    for (entry: (String, Double) <- freqs.toSet) {
      var set = freqMap.get(entry._2)
      if (set == null) {
        set = new mutable.TreeSet[String]() // tokens of the same frequency would be sorted alphabetically
        freqMap.put(entry._2, set)
      }
      set.add(entry._1)
    }

    var cnt = 0
    dictSet ++= dict.keySet
    // get most frequent tokens and put them to dictSet// get most frequent tokens and put them to dictSet

    for (entry: Entry[Double, mutable.TreeSet[String]] <- freqMap.entrySet.asScala) {
      var break = false
      for (value <- entry.getValue if !break) {
        cnt = cnt + 1
        if (dictSet.add(value) && cnt >= MAX_DICT) break = true
        if (cnt >= MAX_DICT) break = true
      }
    }

    // all of the above means that the dictionary with the same MAX_DICT constraint and made from the same source file will always be
    // the same, the tokens always correspond to the same number so we don't need to save/restore the dictionary
    logger.info("Dictionary is ready, size is " + dictSet.size)
    // index the dictionary and build the reverse dictionary for lookups
    for (word <- dictSet) {
      if (!dict.contains(word)) {
        dict.put(word, idx)
        revDict.put(idx, word)
        idx += 1
      }
    }

    logger.info("Total dictionary size is " + dict.size + ". Processing the dataset...")

    corpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, false) {
      override protected def processLine(lastLine: String): Unit = {
        val words = new mutable.HashSet[String]
        tokenizeLine(lastLine, words, addSpecials = true)
        if (words.nonEmpty) {
          val wordIdxs = new ArrayBuffer[Double]
          if (wordsToIndexes(words, wordIdxs)) corpus.append(wordIdxs)
        }
      }
    }
    corpusProcessor.setDict(dict)
    corpusProcessor.start()
    logger.info("Done. Corpus size is " + corpus.size)
  }

  private def toTempPath(path: String) = new ClassPathResource("/encdec").getFile.getPath + "/" + path

  @throws(classOf[IOException])
  private def startDialog(scanner: Scanner) = {
    logger.info("Dialog started.")
    while (true) {
      logger.info("In> ")
      // input line is appended to conform to the corpus format
      val line = "1 ++++++u11++++++ m0 ++++++WALTER++++++ " + scanner.nextLine + "\n"
      val dialogProcessor = new CorpusProcessor(new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), ROW_SIZE,
        false) {
        override protected def processLine(lastLine: String): Unit = {
          val words = new mutable.HashSet[String]
          tokenizeLine(lastLine, words, true)
          val wordIdxs = new ArrayBuffer[Double]
          if (wordsToIndexes(words, wordIdxs)) {
            System.out.print("Got words: ")
            for (idx <- wordIdxs) {
              System.out.print(revDict.get(idx) + " ")
            }
            System.out.println()
            System.out.print("Out> ")
          }
        }
      }
      dialogProcessor.setDict(dict)
      dialogProcessor.start()
    }
  }

  @throws(classOf[IOException])
  private def saveModel(networkFile: File) = {
    logger.info("Saving the model...")
    val backup = new File(toTempPath(BACKUP_MODEL_FILENAME))
    if (networkFile.exists) {
      if (backup.exists) backup.delete
      networkFile.renameTo(backup)
    }
    ModelSerializer.writeModel(net, networkFile, true)
    logger.info("Done.")
  }

  private def test() = {
    logger.info("======================== TEST ========================")
    val selected = rnd.nextInt(corpus.size)
    val old_rowIn: ArrayBuffer[Double] = corpus(selected)
    val rowIn = new ArrayBuffer[Double]()
    rowIn.appendAll(old_rowIn)

    System.out.print("In: ")
    val print_rowIn = rowIn.reverse
    for (idx <- print_rowIn) {
      System.out.print(revDict.getOrElse(idx, "Null") + " ")
    }
    System.out.println()
    System.out.print("Out: ")
    output(rowIn, printUnknowns = true)
    logger.info("====================== TEST END ======================")
  }

  private def createComputationGraph() = {
    val builder = new NeuralNetConfiguration.Builder
    builder.iterations(1).learningRate(LEARNING_RATE).rmsDecay(RMS_DECAY)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).miniBatch(true).updater(Updater.RMSPROP)
      .weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)

    val graphBuilder = builder.graphBuilder().pretrain(false).backprop(true).backpropType(BackpropType.Standard)
      .tBPTTBackwardLength(TBPTT_SIZE).tBPTTForwardLength(TBPTT_SIZE)

    graphBuilder.addInputs("inputLine", "decoderInput")
      .setInputTypes(InputType.recurrent(dict.size), InputType.recurrent(dict.size))
      .addLayer("embeddingEncoder", new EmbeddingLayer.Builder().nIn(dict.size).nOut(EMBEDDING_WIDTH).build, "inputLine")
      .addLayer("encoder", new GravesLSTM.Builder().nIn(EMBEDDING_WIDTH).nOut(HIDDEN_LAYER_WIDTH)
        .activation(Activation.TANH).build, "embeddingEncoder")
      .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
      .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
      .addVertex("merge", new MergeVertex, "decoderInput", "dup")
      .addLayer("decoder", new GravesLSTM.Builder().nIn(dict.size + HIDDEN_LAYER_WIDTH).nOut(HIDDEN_LAYER_WIDTH).activation(Activation.TANH).build, "merge")
      .addLayer("output", new RnnOutputLayer.Builder().nIn(HIDDEN_LAYER_WIDTH).nOut(dict.size).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build, "decoder")
      .setOutputs("output")

    net = new ComputationGraph(graphBuilder.build)
    net.init()
  }

  @throws(classOf[IOException])
  private def train(networkFile: File, offset: Int) = {
    var lastSaveTime = System.currentTimeMillis
    var lastTestTime = System.currentTimeMillis
    val logsIterator = new CorpusIterator(corpus, MINIBATCH_SIZE, MACROBATCH_SIZE, dict.size, ROW_SIZE)

    (1 until 100).foreach{epoch =>
      logger.info("Epoch " + epoch)
      if (epoch == 1) logsIterator.setCurrentBatch(offset)
      else logsIterator.reset()

      var lastPerc = 0
      while (logsIterator.hasNextMacrobatch) {
        net.fit(logsIterator)
        logsIterator.nextMacroBatch()
        logger.info("Batch = " + logsIterator.batch)
        val newPerc = logsIterator.batch * 100 / logsIterator.totalBatches
        if (newPerc != lastPerc) {
          logger.info("Epoch complete: " + newPerc + "%")
          lastPerc = newPerc
        }
        if (System.currentTimeMillis - lastSaveTime > SAVE_EACH_MS) {
          saveModel(networkFile)
          lastSaveTime = System.currentTimeMillis
        }
        if (System.currentTimeMillis - lastTestTime > TEST_EACH_MS) {
          test()
          lastTestTime = System.currentTimeMillis
        }
      }
    }
  }

  private def output(rowIns: ArrayBuffer[Double], printUnknowns: Boolean) = {
    net.rnnClearPreviousState()
    val new_nowIn = rowIns.reverse

    val in = Nd4j.create(new Array[Double](new_nowIn.size), Array[Int](1, 1, new_nowIn.size))
    val decodeArr = new Array[Double](dict.size)
    decodeArr(2) = 1
    var decode = Nd4j.create(decodeArr, Array[Int](1, dict.size, 1))
    net.feedForward(Array[INDArray](in, decode), false)
    val decoder = net.getLayer("decoder").asInstanceOf[org.deeplearning4j.nn.layers.recurrent.GravesLSTM]
    val output = net.getLayer("output")

    val mergeVertex = net.getVertex("merge")
    val thoughtVector = mergeVertex.getInputs()(1)
    var out_break = false
    for (row <- 0 until ROW_SIZE if !out_break) {
      mergeVertex.setInputs(decode, thoughtVector)
      val merged: INDArray = mergeVertex.doForward(false)
      val activateDec: INDArray = decoder.rnnTimeStep(merged)
      val out: INDArray = output.activate(activateDec, false)
      val d: Double = rnd.nextDouble
      var sum: Double = 0.0
      var idx = -1
      var in_break = false
      for (s <- 0 until out.size(1) if !in_break) {
        sum += out.getDouble(0, s, 0)
        if (d <= sum) {
          idx = s
          if (printUnknowns || (s == 0)) System.out.print(revDict.getOrElse(s, "Null") + " ")
          in_break = true
        }
      }
      if (idx == 1) out_break = true
      else {
        val newDecodeArr = new Array[Double](dict.size)
        newDecodeArr(idx) = 1
        decode = Nd4j.create(newDecodeArr, Array[Int](1, dict.size, 1))
      }
    }
    System.out.println()
  }
}

object EncoderDecoderLSTM extends App {
  new EncoderDecoderLSTM().run(args)
}
