package org.dl4scala.examples.nlp.tsne

import java.io.File

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.berkeley
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by endy on 2017/6/25.
  */
object TSNEStandardExample {
  private val log = LoggerFactory.getLogger(TSNEStandardExample.getClass)

  def main(args: Array[String]): Unit = {
    // STEP 1: Initialization
    val iterations = 100
    // create an n-dimensional array of doubles
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    val cacheList = new ArrayBuffer[String](); // cacheList is a dynamic array of strings used to hold all words

    //STEP 2: Turn text input into a list of words
    log.info("Load & Vectorize data....")
    val wordFile = new ClassPathResource("words.txt").getFile //Open the file

    //Get the data of all unique word vectors
    val vectors: berkeley.Pair[InMemoryLookupTable[_ <: SequenceElement], VocabCache[_ <: SequenceElement]] = WordVectorSerializer.loadTxt(wordFile)
    val cache = vectors.getSecond
    val weights = vectors.getFirst.getSyn0 //seperate weights of unique words into their own list

    (0 until cache.numWords()).foreach(i => cacheList.append(cache.wordAtIndex(i)))

    import org.deeplearning4j.plot.BarnesHutTsne
    //STEP 3: build a dual-tree tsne to use later//STEP 3: build a dual-tree tsne to use later

    log.info("Build model....")
    val tsne = new BarnesHutTsne.Builder()
      .setMaxIter(iterations)
      .theta(0.5)
      .normalize(false)
      .learningRate(500)
      .useAdaGrad(false)
      .build

    //STEP 4: establish the tsne values and save them to a file
    log.info("Store TSNE Coordinates for Plotting....")
    val outputFile = "target/archive-tmp/tsne-standard-coords.csv"
    new File(outputFile).getParentFile.mkdirs
    tsne.fit(weights)
    tsne.saveAsFile(cacheList.asJava, outputFile)
  }
}
