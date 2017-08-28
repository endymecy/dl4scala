package org.dl4scala.base

import java.io.{BufferedReader, File, IOException, InputStreamReader}
import java.net.URL

import org.apache.commons.codec.digest.DigestUtils
import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import org.dl4scala.util.ArchiveUtils
import org.slf4j.Logger
import org.slf4j.LoggerFactory


/**
  * Created by endy on 2017/8/25.
  */
class MnistFetcher {
  import MnistFetcher._

  protected var BASE_DIR = new File(System.getProperty("user.home"))
  protected var FILE_DIR = new File(BASE_DIR, LOCAL_DIR_NAME)
  private var fileDir: File = _

  def getName = "MNIST"

  def getBaseDir = FILE_DIR

  // --- Train files ---

  def getTrainingFilesURL: String = trainingFilesURL

  def getTrainingFilesMD5: String = trainingFilesMD5

  def getTrainingFilesFilename: String = trainingFilesFilename

  def getTrainingFilesFilename_unzipped: String = trainingFilesFilename_unzipped

  def getTrainingFileLabelsURL: String = trainingFileLabelsURL

  def getTrainingFileLabelsMD5: String = trainingFileLabelsMD5

  def getTrainingFileLabelsFilename: String = trainingFileLabelsFilename

  def getTrainingFileLabelsFilename_unzipped: String = trainingFileLabelsFilename_unzipped

  // --- Test files ---

  def getTestFilesURL: String = testFilesURL

  def getTestFilesMD5: String = testFilesMD5

  def getTestFilesFilename: String = testFilesFilename

  def getTestFilesFilename_unzipped: String = testFilesFilename_unzipped

  def getTestFileLabelsURL: String = testFileLabelsURL

  def getTestFileLabelsMD5: String = testFileLabelsMD5

  def getTestFileLabelsFilename: String = testFileLabelsFilename

  def getTestFileLabelsFilename_unzipped: String = testFileLabelsFilename_unzipped

  @throws(classOf[IOException])
  def downloadAndUntar(): File = {
    if (fileDir != null) return fileDir
    val baseDir = getBaseDir
    if (!(baseDir.isDirectory || baseDir.mkdir)) throw new IOException("Could not mkdir " + baseDir)

    log.info("Downloading {}...", getName)
    // getFromOrigin training records
    val tarFile = new File(baseDir, getTrainingFilesFilename)
    val testFileLabels = new File(baseDir, getTestFilesFilename)

    tryDownloadingAFewTimes(new URL(getTrainingFilesURL), tarFile, getTrainingFilesMD5)
    tryDownloadingAFewTimes(new URL(getTestFilesURL), testFileLabels, getTestFilesMD5)

    ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath, baseDir.getAbsolutePath)
    ArchiveUtils.unzipFileTo(testFileLabels.getAbsolutePath, baseDir.getAbsolutePath)


    val labels = new File(baseDir, getTrainingFileLabelsFilename)
    val labelsTest = new File(baseDir, getTestFileLabelsFilename)

    tryDownloadingAFewTimes(new URL(getTrainingFileLabelsURL), labels, getTrainingFileLabelsMD5)
    tryDownloadingAFewTimes(new URL(getTestFileLabelsURL), labelsTest, getTestFileLabelsMD5)

    ArchiveUtils.unzipFileTo(labels.getAbsolutePath, baseDir.getAbsolutePath)
    ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath, baseDir.getAbsolutePath)

    fileDir = baseDir
    fileDir
  }

  @throws(classOf[IOException])
  private def tryDownloadingAFewTimes(url: URL, f: File, targetMD5: String): Unit = {
    tryDownloadingAFewTimes(0, url, f, targetMD5)
  }


  @throws(classOf[IOException])
  private def tryDownloadingAFewTimes(attempt: Int, url: URL, f: File, targetMD5: String): Unit = {
    val maxTries = 3
    val isCorrectFile = f.isFile
    if (attempt < maxTries && !isCorrectFile) {
      FileUtils.copyURLToFile(url, f)
      if (!checkMD5OfFile(targetMD5, f)) {
        f.delete
        tryDownloadingAFewTimes(attempt + 1, url, f, targetMD5)
      }
    }
    else if (isCorrectFile) {
      // do nothing, file downloaded
    }
    else throw new IOException("Could not download " + url.getPath + "\n properly despite trying " + maxTries + " times, check your connection. File info:" + "\nTarget MD5: " + targetMD5 + "\nHash matches: " + checkMD5OfFile(targetMD5, f) + "\nIs valid file: " + f.isFile)
  }

  @throws[IOException]
  private def checkMD5OfFile(targetMD5: String, file: File) = {
    val in = FileUtils.openInputStream(file)
    val trueMd5 = DigestUtils.md5Hex(in)
    IOUtils.closeQuietly(in)
    targetMD5 == trueMd5
  }

}

object MnistFetcher {
  protected val log: Logger = LoggerFactory.getLogger(MnistFetcher.getClass)

  protected val LOCAL_DIR_NAME = "MNIST"

  private val trainingFilesURL = "http://benchmark.deeplearn.online/mnist/train-images-idx3-ubyte.gz"
  private val trainingFilesMD5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873"
  private val trainingFilesFilename = "train-images-idx3-ubyte.gz"
  val trainingFilesFilename_unzipped = "train-images-idx3-ubyte"
  private val trainingFileLabelsURL = "http://benchmark.deeplearn.online/mnist/train-labels-idx1-ubyte.gz"
  private val trainingFileLabelsMD5 = "d53e105ee54ea40749a09fcbcd1e9432"
  private val trainingFileLabelsFilename = "train-labels-idx1-ubyte.gz"
  val trainingFileLabelsFilename_unzipped = "train-labels-idx1-ubyte"

  //Test data:
  private val testFilesURL = "http://benchmark.deeplearn.online/mnist/t10k-images-idx3-ubyte.gz"
  private val testFilesMD5 = "9fb629c4189551a2d022fa330f9573f3"
  private val testFilesFilename = "t10k-images-idx3-ubyte.gz"
  val testFilesFilename_unzipped = "t10k-images-idx3-ubyte"
  private val testFileLabelsURL = "http://benchmark.deeplearn.online/mnist/t10k-labels-idx1-ubyte.gz"
  private val testFileLabelsMD5 = "ec29112dd5afa0611ce80d1b7f02629c"
  private val testFileLabelsFilename = "t10k-labels-idx1-ubyte.gz"
  val testFileLabelsFilename_unzipped = "t10k-labels-idx1-ubyte"

  @throws(classOf[IOException])
  def gunzipFile(baseDir: File, gzFile: File): Unit = {
    log.info("gunzip'ing File: " + gzFile.toString)
    val p = Runtime.getRuntime.exec(String.format("gunzip %s", gzFile.getAbsolutePath))
    val stdError = new BufferedReader(new InputStreamReader(p.getErrorStream))
    log.info("Here is the standard error of the command (if any):\n")
    var s: String = null
    while ((s = stdError.readLine) != null)
      log.info(s)
    stdError.close()
  }
}
