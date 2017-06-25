package org.dl4scala.examples.nlp.paragraphvectors.tools

import lombok.NonNull
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.JavaConverters._
import java.util

/**
  * Created by endy on 2017/6/25.
  */
class LabelSeeker(labelsUsed: util.List[String], lookupTable: InMemoryLookupTable[VocabWord]) {
  if (labelsUsed.isEmpty) throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors")

  /**
    * This method accepts vector, that represents any document,
    * and returns distances between this document, and previously trained categories
    */

  def getScores(@NonNull vector: INDArray): util.ArrayList[(String, Double)] = {
    val result = new util.ArrayList[(String, Double)]()
    for (label: String <- labelsUsed.asScala) {
      val vecLabel = lookupTable.vector(label)
      if (vecLabel == null) throw new IllegalStateException("Label '" + label + "' has no known vector!")

      val sim = Transforms.cosineSim(vector, vecLabel)
      result.add((label, sim))
    }
    result
  }
}
