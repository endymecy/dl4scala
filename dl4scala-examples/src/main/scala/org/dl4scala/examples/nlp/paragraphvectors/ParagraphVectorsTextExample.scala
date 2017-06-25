package org.dl4scala.examples.nlp.paragraphvectors

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
import org.deeplearning4j.text.documentiterator.LabelsSource
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.LoggerFactory

/**
  * Created by endy on 2017/6/25.
  */
object ParagraphVectorsTextExample {
  private val log = LoggerFactory.getLogger(ParagraphVectorsTextExample.getClass)

  def main(args: Array[String]): Unit = {

    val resource = new ClassPathResource("/raw_sentences.txt")
    val file = resource.getFile
    val iter = new BasicLineIterator(file)

    val cache = new AbstractCache[VocabWord]()

    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)

    val source = new LabelsSource("DOC_")

    val vec = new ParagraphVectors.Builder()
      .minWordFrequency(1)
      .iterations(5)
      .epochs(1)
      .layerSize(100)
      .learningRate(0.025)
      .labelsSource(source)
      .windowSize(5)
      .iterate(iter)
      .trainWordVectors(false)
      .vocabCache(cache)
      .tokenizerFactory(t)
      .sampling(0)
      .build()

    vec.fit()

    /**
      * In training corpus we have few lines that contain pretty close words invloved.
      *    These sentences should be pretty close to each other in vector space
      *    line 3721: This is my way .
      *    line 6348: This is my case .
      *    line 9836: This is my house .
      *    line 12493: This is my world .
      *    line 16393: This is my work .
      *    this is special sentence, that has nothing common with previous sentences
      *    line 9853: We now have one .
      */

    val similarity1: Double = vec.similarity("DOC_9835", "DOC_12492")
    log.info("9836/12493 ('This is my house .'/'This is my world .') similarity: " + similarity1)

    val similarity2: Double = vec.similarity("DOC_3720", "DOC_16392")
    log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2)

    val similarity3: Double = vec.similarity("DOC_6347", "DOC_3720")
    log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3)

    // likelihood in this case should be significantly lower
    val similarityX: Double = vec.similarity("DOC_3720", "DOC_9852")
    log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX + "(should be significantly lower)")
  }
}
