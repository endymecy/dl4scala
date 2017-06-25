package org.dl4scala.examples.nlp.word2vec

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.LoggerFactory

/**
  * Neural net that processes text into wordvectors
  * Created by endy on 2017/6/25.
  */
object Word2VecRawTextExample {
  private val log = LoggerFactory.getLogger(Word2VecRawTextExample.getClass)

  def main(args: Array[String]): Unit = {
    // Gets Path to Text file
    val filePath: String = new ClassPathResource("raw_sentences.txt").getFile.getAbsolutePath

    log.info("Load & Vectorize Sentences....")
    // Strip white space before and after for each line
    val iter = new BasicLineIterator(filePath)
    // Split on white spaces in the line to get words
    val t = new DefaultTokenizerFactory

    t.setTokenPreProcessor(new CommonPreprocessor)

    import org.deeplearning4j.models.word2vec.Word2Vec

    log.info("Building model....")
    val vec: Word2Vec = new Word2Vec.Builder()
      .minWordFrequency(5)
      .iterations(1)
      .layerSize(100)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(t)
      .build

    log.info("Fitting Word2Vec model....")
    vec.fit()

    log.info("Writing word vectors to text file....")
    // Write word vectors to file
    WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt")
    // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
    log.info("Closest Words:")
    val lst = vec.wordsNearest("day", 10)
    System.out.println("10 Words closest to 'day': " + lst)
  }
}
