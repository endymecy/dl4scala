package org.dl4scala.examples.nlp.paragraphvectors

import java.io.FileNotFoundException

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.text.documentiterator.LabelAwareIterator
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.dl4scala.examples.nlp.paragraphvectors.tools.{LabelSeeker, MeansBuilder}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

/**
  * Created by endy on 2017/6/25.
  */
class ParagraphVectorsClassifierExample {

  private val log = LoggerFactory.getLogger(classOf[ParagraphVectorsClassifierExample])

  var paragraphVectors: ParagraphVectors = _
  var iterator: LabelAwareIterator = _
  var tokenizerFactory: TokenizerFactory = _

  @throws(classOf[Exception])
  def makeParagraphVectors(): Unit = {
    val resource = new ClassPathResource("paravec/labeled")
    // build a iterator for our dataset
    iterator = new FileLabelAwareIterator.Builder()
      .addSourceFolder(resource.getFile).build
    tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)
    // ParagraphVectors training configuration
    paragraphVectors = new ParagraphVectors.Builder()
      .learningRate(0.025)
      .minLearningRate(0.001)
      .batchSize(1000)
      .epochs(20)
      .iterate(iterator)
      .trainWordVectors(true)
      .tokenizerFactory(tokenizerFactory)
      .build
    // Start model training
    paragraphVectors.fit()
  }

  @throws(classOf[FileNotFoundException])
  def checkUnlabeledData(): Unit = {
    /*
        At this point we assume that we have model built and we can check
        which categories our unlabeled document falls into.
        So we'll start loading our unlabeled documents and checking them
       */
    val unClassifiedResource = new ClassPathResource("paravec/unlabeled")
    val unClassifiedIterator = new FileLabelAwareIterator.Builder()
      .addSourceFolder(unClassifiedResource.getFile)
      .build

    /*
       Now we'll iterate over unlabeled data, and check which label it could be assigned to
       Please note: for many domains it's normal to have 1 document fall into few labels at once,
       with different "weight" for each.
    */
    val meansBuilder = new MeansBuilder(paragraphVectors.getLookupTable.asInstanceOf[InMemoryLookupTable[VocabWord]], tokenizerFactory)
    val seeker = new LabelSeeker(iterator.getLabelsSource.getLabels,
      paragraphVectors.getLookupTable.asInstanceOf[InMemoryLookupTable[VocabWord]])

    while (unClassifiedIterator.hasNextDocument) {
      val document = unClassifiedIterator.nextDocument
      val documentAsCentroid = meansBuilder.documentAsVector(document)
      val scores = seeker.getScores(documentAsCentroid)

      /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
      */
      log.info("Document '" + document.getLabel + "' falls into the following categories: ")
      for (score: (String, Double) <- scores.asScala) {
        log.info("        " + score._1 + ": " + score._2)
      }
    }
  }
}

object ParagraphVectorsClassifierExample extends App{
  val app = new ParagraphVectorsClassifierExample
  app.makeParagraphVectors()
  app.checkUnlabeledData()
}
