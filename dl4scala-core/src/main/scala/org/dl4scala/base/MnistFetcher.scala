package org.dl4scala.base

import java.io.{BufferedReader, File, IOException, InputStreamReader}
import java.net.URL

import com.typesafe.scalalogging.LazyLogging
import org.apache.commons.io.FileUtils
import org.dl4scala.util.ArchiveUtils

/**
  * Created by endy on 16-12-20.
  */

object MnistFetcher extends LazyLogging {

  protected val LOCAL_DIR_NAME: String = "MNIST"
  private val trainingFilesURL: String = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  private val trainingFilesFilename: String = "images-idx3-ubyte.gz"
  val trainingFilesFilename_unzipped: String = "images-idx3-ubyte"
  private val trainingFileLabelsURL: String = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  private val trainingFileLabelsFilename: String = "labels-idx1-ubyte.gz"
  val trainingFileLabelsFilename_unzipped: String = "labels-idx1-ubyte"

  //Test data:
  private val testFilesURL: String = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
  private val testFilesFilename: String = "t10k-images-idx3-ubyte.gz"
  val testFilesFilename_unzipped: String = "t10k-images-idx3-ubyte"
  private val testFileLabelsURL: String = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
  private val testFileLabelsFilename: String = "t10k-labels-idx1-ubyte.gz"
  val testFileLabelsFilename_unzipped: String = "t10k-labels-idx1-ubyte"

  protected val BASE_DIR: File = new File(System.getProperty("user.home"))
  protected val FILE_DIR: File = new File(BASE_DIR, LOCAL_DIR_NAME)
  private var fileDir: File = _


  @throws[IOException]
  def downloadAndUntar: File = {
    if (fileDir != null)  return fileDir

    val baseDir: File = FILE_DIR
    if (!(baseDir.isDirectory || baseDir.mkdir))
      throw new IOException("Could not mkdir " + baseDir)

    logger.info("Downloading mnist...")

    // getFromOrigin training records
    val tarFile: File = new File(baseDir, trainingFilesFilename)
    val tarFileLabels: File = new File(baseDir, testFilesFilename)
    if (!tarFile.isFile) FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile)
    if (!tarFileLabels.isFile) FileUtils.copyURLToFile(new URL(testFilesURL), tarFileLabels)
    ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath, baseDir.getAbsolutePath)
    ArchiveUtils.unzipFileTo(tarFileLabels.getAbsolutePath, baseDir.getAbsolutePath)

    // getFromOrigin training records
    val labels: File = new File(baseDir, trainingFileLabelsFilename)
    val labelsTest: File = new File(baseDir, testFileLabelsFilename)
    if (!labels.isFile) FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), labels)
    if (!labelsTest.isFile) FileUtils.copyURLToFile(new URL(testFileLabelsURL), labelsTest)
    ArchiveUtils.unzipFileTo(labels.getAbsolutePath, baseDir.getAbsolutePath)
    ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath, baseDir.getAbsolutePath)

    fileDir = baseDir
    fileDir
  }

  @throws(classOf[IOException])
  def gunzipFile(baseDir: File, gzFile: File) {
    logger.info("gunzip'ing File: " + gzFile.toString)

    val p: Process = Runtime.getRuntime.exec(String.format("gunzip %s", gzFile.getAbsolutePath))
    val stdError: BufferedReader = new BufferedReader(new InputStreamReader(p.getErrorStream))

    logger.info("Here is the standard error of the command (if any):\n")
    Stream.continually(stdError.readLine()).takeWhile(_ != null).foreach(x => logger.info(x))
    stdError.close()
  }
}
