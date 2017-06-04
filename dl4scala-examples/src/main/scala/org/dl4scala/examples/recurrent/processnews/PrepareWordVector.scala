package org.dl4scala.examples.recurrent.processnews

import java.io.File

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.io.ClassPathResource
import org.slf4j.LoggerFactory

/**
  * Created by endy on 2017/6/4.
  */
object PrepareWordVector extends App{
  private val log = LoggerFactory.getLogger(classOf[PrepareWordVector.type])

  // Gets Path to Text file
  val classPathResource = new ClassPathResource("NewsData").getFile.getAbsolutePath + File.separator
  val filePath = new File(classPathResource + File.separator + "RawNewsToGenerateWordVector.txt").getAbsolutePath

  log.info("Load & Vectorize Sentences....")
  // Strip white space before and after for each line
  val iter = new BasicLineIterator(filePath)
  // Split on white spaces in the line to get words
  val t = new DefaultTokenizerFactory

  // CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
  // So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
  // Additionally it forces lower case for all tokens.
  t.setTokenPreProcessor(new CommonPreprocessor)

  log.info("Building model....")
  val vec = new Word2Vec.Builder()
    .minWordFrequency(2)
    .iterations(5)
    .layerSize(100)
    .seed(42)
    .windowSize(20)
    .iterate(iter)
    .tokenizerFactory(t)
    .build

  log.info("Fitting Word2Vec model....")
  vec.fit()

  log.info("Writing word vectors to text file....")

  // Write word vectors to file
  WordVectorSerializer.writeWordVectors(vec.getLookupTable, classPathResource + "NewsWordVector.txt")

}
