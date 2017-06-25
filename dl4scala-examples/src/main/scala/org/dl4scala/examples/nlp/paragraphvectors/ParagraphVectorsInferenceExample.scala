package org.dl4scala.examples.nlp.paragraphvectors

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.LoggerFactory

/**
  * This is example code for dl4j ParagraphVectors inference use implementation.
  * In this example we load previously built model, and pass raw sentences, probably never seen before,
  * to get their vector representation.
  *
  * Created by endy on 2017/6/25.
  */
object ParagraphVectorsInferenceExample {
  private val log = LoggerFactory.getLogger(ParagraphVectorsInferenceExample.getClass)

  def main(args: Array[String]): Unit = {
    val resource = new ClassPathResource("/paravec/simple.pv")
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)

    // we load externally originated model
    val vectors = WordVectorSerializer.readParagraphVectors(resource.getFile)
    vectors.setTokenizerFactory(t)
    vectors.getConfiguration.setIterations(1) // please note, we set iterations to 1 here, just to speedup inference

    val inferredVectorA = vectors.inferVector("This is my world .")
    val inferredVectorA2 = vectors.inferVector("This is my world .")
    val inferredVectorB = vectors.inferVector("This is my way .")

    // high similarity expected here, since in underlying corpus words WAY and WORLD have really close context
    log.info("Cosine similarity A/B: {}", Transforms.cosineSim(inferredVectorA, inferredVectorB))

    // equality expected here, since inference is happening for the same sentences
    log.info("Cosine similarity A/A2: {}", Transforms.cosineSim(inferredVectorA, inferredVectorA2))
  }
}
