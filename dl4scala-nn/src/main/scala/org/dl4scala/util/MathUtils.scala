package org.dl4scala.util

import scala.util.Random

/**
  * Created by endy on 2017/8/28.
  */
object MathUtils {
  def shuffleArray(array: Array[Int], rngSeed: Long): Unit = {
    shuffleArray(array, new Random(rngSeed))
  }

  def shuffleArray(array: Array[Int], rng: Random): Unit = {
    //https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
    var i = array.length - 1
    while (i > 0) {
      val j = rng.nextInt(i + 1)
      val temp = array(j)
      array(j) = array(i)
      array(i) = temp
      i -= 1
    }
  }
}
