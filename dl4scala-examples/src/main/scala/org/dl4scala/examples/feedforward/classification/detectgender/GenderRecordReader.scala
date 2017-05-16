package org.dl4scala.examples.feedforward.classification.detectgender

import org.datavec.api.records.reader.impl.LineRecordReader
import java.io.{File, IOException}
import java.net.URI
import java.nio.file.{Files, Paths}
import java.nio.charset.Charset

import org.apache.commons.lang3.StringUtils
import org.datavec.api.conf.Configuration
import org.datavec.api.split.{FileSplit, InputSplit, InputStreamInputSplit}
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer


/**
  * Created by endy on 2017/5/13.
  */
class GenderRecordReader(labels: List[String]) extends LineRecordReader {

  private val log: Logger = LoggerFactory.getLogger(classOf[GenderRecordReader])

  // Final list that contains actual binary data generated from person name, it also contains label (1 or 0) at the end
  private val names = new ArrayBuffer[String]

  // String to hold all possible alphabets from all person names in raw data
  // This String is used to convert person name to binary string seperated by comma
  private var possibleCharacters = ""

  // holds length of largest name out of all person names
  var maxLengthName = 0

  // holds total number of names including both male and female names.
  // This variable is not used in PredictGenderTrain.java
  private var _totalRecords = 0

  // iterator for List "names" to be used in next() method
  private var iter: scala.collection.Iterator[String] = _

  /**
    * returns total number of records in List "names"
    *
    * @return - totalRecords
    */
  private def totalRecords = _totalRecords

  /**
    * This function does following steps
    * - Looks for the files with the name (in specified folder) as specified in labels set in constructor
    * - File must have person name and gender of the person (M or F),
    *   e.g. Deepan,M
    * Trupesh,M
    * Vinay,M
    * Ghanshyam,M
    *
    * Meera,F
    * Jignasha,F
    * Chaku,F
    *
    * - File for male and female names must be different, like M.csv, F.csv etc.
    * - populates all names in temporary list
    * - generate binary string for each alphabet for all person names
    * - combine binary string for all alphabets for each name
    * - find all unique alphabets to generate binary string mentioned in above step
    * - take equal number of records from all files. To do that, finds minimum record from all files, and then takes
    * that number of records from all files to keep balance between data of different labels.
    * - Note : this function uses stream() feature of Java 8, which makes processing faster. Standard method to process file takes more than 5-7 minutes.
    * using stream() takes approximately 800-900 ms only.
    * - Final converted binary data is stored List<String> names variable
    * - sets iterator from "names" list to be used in next() function
    *
    * @param split - user can pass directory containing .CSV file for that contains names of male or female
    * @throws IOException IOException
    * @throws InterruptedException InterruptedException
    */
  @throws(classOf[IOException])
  @throws(classOf[InterruptedException])
  override def initialize(split: InputSplit): Unit = {
    split match {
      case _: FileSplit =>
        val locations = split.locations
        if (locations != None && locations.length > 1) {
          var longestName = ""
          var uniqueCharactersTempString = ""
          val tempNames = new ArrayBuffer[(String, ArrayBuffer[String])]()
          for (location: URI <- locations) {
            val file = new File(location)
            val temp = this.labels.filter(line => file.getName.equals(line + ".csv"))

            if (temp.nonEmpty) {
              val path = Paths.get(file.getAbsolutePath)
              val tempList = Files.readAllLines(path, Charset.defaultCharset()).toArray().map { case element: String => element.split(",")(0) }
              longestName = tempList.reduce((name1, name2) => if (name1.length >= name2.length) name1 else name2)

              uniqueCharactersTempString = uniqueCharactersTempString + tempList.toString
              val tempPair = (temp.head, tempList.to[ArrayBuffer])
              tempNames.append(tempPair)
            } else throw new InterruptedException("File missing for any of the specified labels")
          }

          this.maxLengthName = longestName.length

          var unique = uniqueCharactersTempString
            .distinct
            .sorted
            .replaceAll("\\[", "")
            .replaceAll("\\]", "")
            .replaceAll(",", "")

          if (unique.startsWith(" ")) unique = " " + unique.trim

          this.possibleCharacters = unique

          val minSize = tempNames.toArray.map(x => x._2.size).min

          val oneMoreTempNames = new ArrayBuffer[(String, ArrayBuffer[String])]()

          for (i <- tempNames.indices) {
            val diff = Math.abs(minSize - tempNames(i)._2.size)
            var tempList = new ArrayBuffer[String]()
            if (tempNames(i)._2.size > minSize) {
              tempList = tempNames(i)._2
              tempList = tempList.slice(0, tempList.size - diff)
            }
            else tempList = tempNames(i)._2

            val tempNewPair = (tempNames(i)._1, tempList)
            oneMoreTempNames.append(tempNewPair)
          }

          tempNames.clear

          val secondMoreTempNames = new ArrayBuffer[(String, ArrayBuffer[String])]()

          for (i <- oneMoreTempNames.indices) {
            val gender = if (oneMoreTempNames(i)._1.equals("M")) 1 else 0
            val secondList = oneMoreTempNames(i)._2.map(element => getBinaryString(element.split(",")(0), gender))
            val secondTempPair = (oneMoreTempNames(i)._1, secondList)
            secondMoreTempNames.append(secondTempPair)
          }

          oneMoreTempNames.clear

          for (i <- secondMoreTempNames.indices) {
            names.appendAll(secondMoreTempNames(i)._2)
          }

          secondMoreTempNames.clear
          this._totalRecords = names.size
          //Collections.shuffle(names)
          this.iter = names.iterator
        }
        else throw new InterruptedException("File missing for any of the specified labels")
      case _: InputStreamInputSplit =>
        log.info("InputStream Split found...Currently not supported")
        throw new InterruptedException("File missing for any of the specified labels")
      case _ =>
    }
  }

  /**
    * - takes onme record at a time from names list using iter iterator
    * - stores it into Writable List and returns it
    * @return
    */
  override def next: List[Writable] =
    if (iter.hasNext) {
      val ret = new ArrayBuffer[Writable]
      val currentRecord = iter.next
      val temp = currentRecord.split(",")
      var i = 0
      while (i < temp.length) {
        ret.append(new DoubleWritable(temp(i).toDouble))
        i = i + 1
      }
      ret.toList
    } else throw new IllegalStateException("no more elements")

  override def hasNext: Boolean = {
    if (iter != null) iter.hasNext
    throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist")
  }


  @throws(classOf[IOException])
  override def close(): Unit = {
  }

  override def setConf(conf: Configuration): Unit = {
    this.conf = conf
  }

  override def getConf: Configuration = conf

  override def reset(): Unit = {
    this.iter = names.iterator
  }

  /**
    * This function gives binary string for full name string
    * - It uses "PossibleCharacters" string to find the decimal equivalent to any alphabet from it
    * - generate binary string for each alphabet
    * - left pads binary string for each alphabet to make it of size 5
    * - combine binary string for all alphabets of a name
    * - Right pads complete binary string to make it of size that is the size of largest name to keep all name length of equal size
    * - appends label value (1 or 0 for male or female respectively)
    * @param name - person name to be converted to binary string
    * @param gender - variable to decide value of label to be added to name's binary string at the end of the string
    * @return
    */
  private def getBinaryString(name: String, gender: Int) = {
    var binaryString = ""
    var j = 0
    while ( j < name.length) {
      val fs = StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))), 5, "0")
      binaryString = binaryString + fs
      j = j + 1
    }
    binaryString = StringUtils.rightPad(binaryString, this.maxLengthName * 5, "0")
    binaryString = binaryString.replaceAll(".(?!$)", "$0,")
    binaryString + "," + String.valueOf(gender)
  }

}

