package org.dl4scala.examples.recurrent.encdec

import java.io._
import java.nio.charset.StandardCharsets
import java.util

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by endy on 2017/6/3.
  */
class CorpusProcessor( is: InputStream, rowSize: Int, countFreq: Boolean) {
  val SPECIALS = "!\"#$;%^:?*()[]{}<>«»,.–—=+…"
  private val dictSet = new mutable.HashSet[String]()
  private val freq = new mutable.OpenHashMap[String, Double]()
  private var dict = new mutable.OpenHashMap[String, Double]()

  @throws(classOf[FileNotFoundException])
  def this(filename: String, rowSize: Int, countFreq: Boolean) {
    this(new FileInputStream(filename), rowSize, countFreq)
  }

  @throws(classOf[IOException])
  def start(): Unit = {
    val br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))
    var lastName = ""
    var lastLine = ""
    Stream.continually(br.readLine()).takeWhile(line => line != null).foreach{line =>
      val lineSplit = line.toLowerCase.split(" \\+\\+\\+\\$\\+\\+\\+ ", 5)
      if (lineSplit.length > 4) {
        // join consecuitive lines from the same sp
        if (lineSplit(1).equals(lastName)) {
          if (!lastLine.isEmpty) {
            // if the previous line doesn't end with a special symbol, append a comma and the current line
            if (!SPECIALS.contains(lastLine.substring(lastLine.length - 1))) lastLine = lastLine + ","
            lastLine = lastLine + " " + lineSplit(4)
          } else lastLine = lineSplit(4)
        } else{
          if (lastLine.isEmpty) lastLine = lineSplit(4)
          else {
            processLine(lastLine)
            lastLine = lineSplit(4)
          }
          lastName = lineSplit(1)
        }
      }
    }
    processLine(lastLine)
    br.close()
  }

  protected def processLine(lastLine: String): Unit = {
    tokenizeLine(lastLine, dictSet, addSpecials = false)
  }

  // here we not only split the words but also store punctuation marks
  protected def tokenizeLine(lastLine: String, resultCollection: mutable.HashSet[String],
                             addSpecials: Boolean): Unit = {
    val words = lastLine.split("[ \t]")
    for (realword <- words) {
      if (!realword.isEmpty) {
        var specialFound = true
        var word = realword
        var break = false
        while (specialFound && !word.isEmpty && !break) {
          (0 until word.length).foreach{i =>
            val idx: Int = SPECIALS.indexOf(word.charAt(i))
            specialFound = false
            if (idx >= 0) {
              val word1 = word.substring(0, i)
              if (!word1.isEmpty) addWord(resultCollection, word1)
              if (addSpecials) addWord(resultCollection, String.valueOf(word.charAt(i)))
              word = word.substring(i + 1)
              specialFound = true
              break = true
            }
          }
        }
        if (!word.isEmpty) addWord(resultCollection, word)
      }
    }
  }

  private def addWord(coll: mutable.HashSet[String], word: String) = {
    if (coll != null) coll.add(word)
    if (countFreq) {
      val count: Double = freq.getOrElse(word, null)
      if (count == null) freq.put(word, 1.0)
      else freq.put(word, count + 1)
    }
  }

  def getDictSet: mutable.HashSet[String] = dictSet

  def getFreq: mutable.OpenHashMap[String, Double] = freq

  def setDict(dict: mutable.OpenHashMap[String, Double]): Unit = {
    this.dict = dict
  }

  protected def wordsToIndexes(words: util.Collection[String], wordIdxs: ArrayBuffer[Double]): Boolean = {
    var i = rowSize
    var break: Boolean = false
    for (word <- words if break) {
      i = i - 1
      if(i == 0) break = true
      val wordIdx: Double = dict.getOrElse(word, null)
      if (wordIdx != null) wordIdxs.append(wordIdx)
      else wordIdxs.append(0.0)
    }
    if (wordIdxs.nonEmpty)  true
    false
  }
}
