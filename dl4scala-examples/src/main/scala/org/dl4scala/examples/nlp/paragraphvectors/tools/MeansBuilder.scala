package org.dl4scala.examples.nlp.paragraphvectors.tools

import java.util.concurrent.atomic.AtomicInteger

import lombok.NonNull
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.text.documentiterator.LabelledDocument
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/25.
  */
class MeansBuilder(lookupTable: InMemoryLookupTable[VocabWord], tokenizerFactory: TokenizerFactory) {

  private val vocabCache = lookupTable.getVocab

  /**
    * This method returns centroid (mean vector) for document.
    *
    * @param document
    */
  def documentAsVector(@NonNull document: LabelledDocument): INDArray = {
    val documentAsTokens = tokenizerFactory.create(document.getContent).getTokens.asScala
    val cnt = new AtomicInteger(0)

    for (word <- documentAsTokens) {
      if (vocabCache.containsWord(word)) cnt.incrementAndGet
    }

    val allWords = Nd4j.create(cnt.get, lookupTable.layerSize)
    cnt.set(0)

    for (word <- documentAsTokens) {
      if (vocabCache.containsWord(word)) allWords.putRow(cnt.getAndIncrement, lookupTable.vector(word))
    }

    val mean = allWords.mean(0)

    mean
  }
}
