package org.dl4scala.examples.nlp.word2vec

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.{VocabWord, Word2Vec}
import org.deeplearning4j.models.word2vec.wordstore.inmemory.{AbstractCache, InMemoryLookupCache}
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.LoggerFactory

/**
  * Created by endy on 2017/6/25.
  */
object Word2VecUptrainingExample {
  private val log = LoggerFactory.getLogger(Word2VecUptrainingExample.getClass)

  def main(args: Array[String]): Unit = {
    val filePath = new ClassPathResource("raw_sentences.txt").getFile.getAbsolutePath

    log.info("Load & Vectorize Sentences....")
    // Strip white space before and after for each line
    val iter = new BasicLineIterator(filePath)
    // Split on white spaces in the line to get words
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)


    // manual creation of VocabCache and WeightLookupTable usually isn't necessary
    // but in this case we'll need them
    val cache = new AbstractCache[VocabWord]()
    val table = new InMemoryLookupTable.Builder[VocabWord]()
      .vectorLength(100)
      .useAdaGrad(false)
      .cache(cache)
      .build

    log.info("Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(5)
      .iterations(1)
      .epochs(1)
      .layerSize(100)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(t)
      .lookupTable(table)
      .vocabCache(cache)
      .build

    log.info("Fitting Word2Vec model....")
    vec.fit()

    var lst = vec.wordsNearest("day", 10)
    log.info("Closest words to 'day' on 1st run: " + lst)

    WordVectorSerializer.writeWord2VecModel(vec, "pathToSaveModel.txt")

    val word2Vec = WordVectorSerializer.readWord2VecModel("pathToSaveModel.txt")

    val iterator = new BasicLineIterator(filePath)
    val tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

    word2Vec.setTokenizerFactory(tokenizerFactory)
    word2Vec.setSentenceIterator(iterator)


    log.info("Word2vec uptraining...")

    word2Vec.fit()

    lst = word2Vec.wordsNearest("day", 10)
    log.info("Closest words to 'day' on 2nd run: " + lst)
  }
}
