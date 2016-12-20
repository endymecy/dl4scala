package org.dl4scala.util

import java.io._
import java.util.zip.{GZIPInputStream, ZipEntry, ZipInputStream}

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, IOUtils}
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by endy on 16-12-20.
  */
case class ArchiveUtils()

object ArchiveUtils {
  private val log: Logger = LoggerFactory.getLogger(classOf[ArchiveUtils])

  /**
    * Extracts files to the specified destination
    *
    * @param file the file to extract to
    * @param dest the destination directory
    * @throws IOException
    */
  @throws[IOException]
  def unzipFileTo(file: String, dest: String) {

    val target: File = new File(file)
    if (!target.exists) throw new IllegalArgumentException("Archive doesnt exist")
    val fin: FileInputStream = new FileInputStream(target)
    val BUFFER: Int = 2048
    val data: Array[Byte] = new Array[Byte](BUFFER)

    if (file.endsWith(".zip")) {
      //getFromOrigin the zip file content
      val zis: ZipInputStream = new ZipInputStream(fin)
      //getFromOrigin the zipped file list entry
      var ze: ZipEntry = zis.getNextEntry
      while (ze != null) {
        val fileName: String = ze.getName
        val newFile: File = new File(dest + File.separator + fileName)
        log.info("file unzip : " + newFile.getAbsoluteFile)
        //createComplex all non exists folders
        //else you will hit FileNotFoundException for compressed folder
        new File(newFile.getParent).mkdirs
        val fos: FileOutputStream = new FileOutputStream(newFile)

        Stream.continually(zis.read(data)).takeWhile(_ > 0).foreach{len => fos.write(data, 0,
          len)}

        fos.close()
        ze = zis.getNextEntry
      }
      zis.closeEntry()
      zis.close()
    }
    else if (file.endsWith(".tar.gz") || file.endsWith(".tgz")) {
      val in: BufferedInputStream = new BufferedInputStream(fin)
      val gzIn: GzipCompressorInputStream = new GzipCompressorInputStream(in)
      val tarIn: TarArchiveInputStream = new TarArchiveInputStream(gzIn)

      /** Read the tar entries using the getNextEntry method **/
      Stream.continually(tarIn.getNextEntry).takeWhile(_ != null).foreach {entry =>
        log.info("Extracting: " + entry.getName)
        /** If the entry is a directory, createComplex the directory. **/
        if (entry.isDirectory) {
          val f: File = new File(dest + File.separator + entry.getName)
          f.mkdirs
        } else {
        /**
          * If the entry is a file,write the decompressed file to the disk
          * and close destination stream.
          **/
          val fos: FileOutputStream = new FileOutputStream(dest + File.separator + entry.getName)
          val destStream: BufferedOutputStream = new BufferedOutputStream(fos, BUFFER)
          Stream.continually(tarIn.read(data, 0, BUFFER)).takeWhile(_ != -1).foreach {count =>
            destStream.write(data, 0, count)
          }
          destStream.flush()
          IOUtils.closeQuietly(destStream)
        }
      }
      /** Close the input stream **/
      tarIn.close()
    }
    else if (file.endsWith(".gz")) {
      val is2: GZIPInputStream = new GZIPInputStream(fin)
      val extracted: File = new File(target.getParent, target.getName.replace(".gz", ""))
      if (extracted.exists) extracted.delete
      extracted.createNewFile
      val fos: OutputStream = FileUtils.openOutputStream(extracted)
      IOUtils.copyLarge(is2, fos)
      is2.close()
      fos.flush()
      fos.close()
    }
    target.delete
  }
}
