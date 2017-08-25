package org.dl4scala.examples.inference

import org.deeplearning4j.parallelism.ParallelInference
import org.deeplearning4j.parallelism.inference.InferenceMode

import org.deeplearning4j.util.ModelSerializer

/**
  * Created by endy on 2017/8/25.
  */
object ParallelInferenceExample {
  def main(args: Array[String]) : Unit = {
    val model = ModelSerializer.restoreComputationGraph("PATH_TO_YOUR_MODEL_FILE", false)

    val pi = new ParallelInference.Builder(model)
      // BATCHED mode is kind of optimization: if number of incoming requests is too high - PI will be batching individual queries into single batch. If number of requests will be low - queries will be processed without batching
      .inferenceMode(InferenceMode.BATCHED)
      // max size of batch for BATCHED mode. you should set this value with respect to your environment (i.e. gpu memory amounts)
      .batchLimit(32)
      // set this value to number of available computational devices, either CPUs or GPUs
      .workers(2)
      .build()

    val result = pi.output(Array[Float](0.1f, 0.1f, 0.1f, 0.2f, 0, 3f))
  }
}
