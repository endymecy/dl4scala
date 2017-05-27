package org.dl4scala.examples.utilities

import java.io._

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by endy on 2017/5/26.
  */
object DataUtilities {
  val logger: Logger = LoggerFactory.getLogger(DataUtilities.getClass)
  private val BUFFER_SIZE = 4096

  @throws(classOf[IOException])
  def extractTarGz(filePath: String, outputPath: String): Unit = {
    var fileCount = 0
    var dirCount = 0

    logger.info("Extracting files")

    val tais = new TarArchiveInputStream(new GzipCompressorInputStream(
      new BufferedInputStream(new FileInputStream(filePath))))
    // Read the tar entries using the getNextEntry method
    Stream.continually(tais.getNextTarEntry).takeWhile(_ !=null).foreach{ entry =>
      // Create directories as required
      if (entry.isDirectory) {
        new File(outputPath + "/" + entry.getName).mkdirs
        dirCount += 1
      } else {
        val data = new Array[Byte](BUFFER_SIZE)
        val fos = new FileOutputStream(outputPath + "/" + entry.getName)
        val dest = new BufferedOutputStream(fos, BUFFER_SIZE)
        Stream.continually(tais.read(data, 0, BUFFER_SIZE)).takeWhile(_ != -1).foreach{ count =>
          dest.write(data, 0, count)
        }
        dest.close()
        fileCount = fileCount + 1
      }
      if (fileCount % 1000 == 0) logger.info(".")
    }

    tais.close()
  }
}
