package org.dl4scala.examples.nlp.sequencevectors

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.{Logger, LoggerFactory}

/**
  * This is example of abstract sequence of data is learned using SequenceVectors.
  * In this example, we use text sentences as Sequences, and VocabWords as SequenceElements.
  * So, this example is  a simple demonstration how one can learn distributed representation of data sequences.
  *
  * For training on different data, you can extend base class SequenceElement, and feed model with your Iterable.
  * Aslo, please note, in this case model persistence should be handled on your side.
  * Created by endy on 2017/6/25.
  */
object SequenceVectorsTextExample {
  protected val logger: Logger = LoggerFactory.getLogger(SequenceVectorsTextExample.getClass)

  def main(args: Array[String]): Unit = {
    val file = new ClassPathResource("raw_sentences.txt").getFile

    val vocabCache = new AbstractCache.Builder[VocabWord]().build

    val underlyingIterator = new BasicLineIterator(file)

    // Now we need the way to convert lines into Sequences of VocabWords.
    // In this example that's SentenceTransformer
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)

    val transformer = new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build

    val sequenceIterator = new AbstractSequenceIterator.Builder(transformer).build()

    val constructor = new VocabConstructor.Builder[VocabWord]()
      .addSource(sequenceIterator, 5)
      .setTargetVocabCache(vocabCache)
      .build()

    constructor.buildJointVocabulary(false, true)

    val lookupTable = new InMemoryLookupTable.Builder[VocabWord]()
      .lr(0.025)
      .vectorLength(150)
      .useAdaGrad(false)
      .cache(vocabCache)
      .build()

    lookupTable.resetWeights(true)

    val vectors = new SequenceVectors.Builder[VocabWord](new VectorsConfiguration())
      // minimum number of occurencies for each element in training corpus. All elements below this value will be ignored
      // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
      .minWordFrequency(5)
      // WeightLookupTable
      .lookupTable(lookupTable)
      // abstract iterator that covers training corpus
      .iterate(sequenceIterator)
      // vocabulary built prior to modelling
      .vocabCache(vocabCache)
      // batchSize is the number of sequences being processed by 1 thread at once
      // this value actually matters if you have iterations > 1
      .batchSize(250)
      // number of iterations over batch
      .iterations(1)
      // number of iterations over whole training corpus
      .epochs(1)
      // if set to true, vocabulary will be built from scratches internally
      // otherwise externally provided vocab will be used
      .resetModel(false)
      // These two methods define our training goals. At least one goal should be set to TRUE.
      .trainElementsRepresentation(true)
      .trainSequencesRepresentation(false)
      // Specifies elements learning algorithms. SkipGram, for example.
      .elementsLearningAlgorithm(new SkipGram[VocabWord]())
      .build()

    vectors.fit()

    val sim = vectors.similarity("day", "night")
    logger.info("Day/night similarity: " + sim)
  }
}
